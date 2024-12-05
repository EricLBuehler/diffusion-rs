use std::{cmp::Ordering, collections::HashMap, fs, sync::Arc};

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::Module;
use serde::Deserialize;
use tokenizers::{models::bpe::BPE, ModelWrapper, Tokenizer};

use crate::{
    models::{
        dispatch_load_vae_model, ClipTextConfig, ClipTextTransformer, FluxConfig, FluxModel,
        T5Config, T5EncoderModel, VAEModel,
    },
    util::from_mmaped_safetensors,
};

use super::{ComponentElem, DiffusionGenerationParams, Loader, ModelPipeline};

mod sampling;

pub struct FluxLoader;

#[derive(Clone, Debug, Deserialize)]
struct SchedulerConfig {
    base_image_seq_len: usize,
    base_shift: f64,
    max_image_seq_len: usize,
    max_shift: f64,
    shift: f64,
    use_dynamic_shifting: bool,
}

impl Loader for FluxLoader {
    fn required_component_names(&self) -> Vec<&'static str> {
        vec![
            "scheduler",
            "text_encoder",
            "text_encoder_2",
            "tokenizer",
            "tokenizer_2",
            "transformer",
            "vae",
        ]
    }

    fn load_from_components(
        &self,
        components: HashMap<String, ComponentElem>,
        device: &Device,
    ) -> Result<Arc<dyn ModelPipeline>> {
        let scheduler = components["scheduler"].clone();
        let clip_component = components["text_encoder"].clone();
        let t5_component = components["text_encoder_2"].clone();
        let clip_tok_component = components["tokenizer"].clone();
        let t5_tok_component = components["tokenizer_2"].clone();
        let flux_component = components["transformer"].clone();
        let vae_component = components["vae"].clone();

        let scheduler_config = if let ComponentElem::Config { files } = scheduler {
            serde_json::from_str::<SchedulerConfig>(&fs::read_to_string(
                files["scheduler/scheduler_config.json"].clone(),
            )?)?
        } else {
            anyhow::bail!("expected scheduler config")
        };
        let clip_tokenizer = if let ComponentElem::Other { files } = clip_tok_component {
            let vocab_file = files["tokenizer/vocab.json"].clone();
            let merges_file = files["tokenizer/merges.txt"].clone();
            let vocab: HashMap<String, u32> =
                serde_json::from_str(&fs::read_to_string(vocab_file)?)?;
            let merges: Vec<(String, String)> = fs::read_to_string(merges_file)?
                .split('\n')
                .skip(1)
                .map(|x| x.split(' ').collect::<Vec<_>>())
                .filter(|x| x.len() == 2)
                .map(|x| (x[0].to_string(), x[1].to_string()))
                .collect();

            Tokenizer::new(ModelWrapper::BPE(BPE::new(vocab, merges)))
        } else {
            anyhow::bail!("incorrect storage of clip tokenizer")
        };
        let t5_tokenizer = if let ComponentElem::Other { files } = t5_tok_component {
            Tokenizer::from_file(files["tokenizer_2/tokenizer.json"].clone())
                .map_err(anyhow::Error::msg)?
        } else {
            anyhow::bail!("incorrect storage of t5 tokenizer")
        };
        let clip_component = if let ComponentElem::Model {
            safetensors,
            config,
        } = clip_component
        {
            let cfg: ClipTextConfig = serde_json::from_str(&fs::read_to_string(config)?)?;
            let vb = from_mmaped_safetensors(
                safetensors.values().cloned().collect(),
                None,
                device,
                false,
            )?;
            ClipTextTransformer::new(vb.pp("text_model"), &cfg)?
        } else {
            anyhow::bail!("incorrect storage of clip model")
        };
        let t5_component = if let ComponentElem::Model {
            safetensors,
            config,
        } = t5_component
        {
            let cfg: T5Config = serde_json::from_str(&fs::read_to_string(config)?)?;
            let vb = from_mmaped_safetensors(
                safetensors.values().cloned().collect(),
                None,
                device,
                false,
            )?;
            T5EncoderModel::new(vb, &cfg, device)?
        } else {
            anyhow::bail!("incorrect storage of t5 model")
        };
        let vae_component = if let ComponentElem::Model {
            safetensors,
            config,
        } = vae_component
        {
            dispatch_load_vae_model(&config, safetensors.values().cloned().collect(), device)?
        } else {
            anyhow::bail!("incorrect storage of vae model")
        };
        let flux_component = if let ComponentElem::Model {
            safetensors,
            config,
        } = flux_component
        {
            let cfg: FluxConfig = serde_json::from_str(&fs::read_to_string(config)?)?;
            let vb = from_mmaped_safetensors(
                safetensors.values().cloned().collect(),
                None,
                device,
                false,
            )?;
            FluxModel::new(&cfg, vb, device)?
        } else {
            anyhow::bail!("incorrect storage of flux model")
        };

        let pipeline = FluxPipeline {
            clip_tokenizer,
            clip_model: clip_component,
            t5_tokenizer,
            t5_model: t5_component,
            vae_model: vae_component,
            flux_model: flux_component,
            scheduler_config,
        };

        Ok(Arc::new(pipeline))
    }
}

pub struct FluxPipeline {
    clip_tokenizer: Tokenizer,
    clip_model: ClipTextTransformer,
    t5_tokenizer: Tokenizer,
    t5_model: T5EncoderModel,
    vae_model: Arc<dyn VAEModel>,
    flux_model: FluxModel,
    scheduler_config: SchedulerConfig,
}

impl ModelPipeline for FluxPipeline {
    fn forward(
        &self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
    ) -> candle_core::Result<Tensor> {
        let mut t5_input_ids = Tensor::new(
            self.t5_tokenizer
                .encode_batch(prompts.clone(), true)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                .into_iter()
                .map(|e| e.get_ids().to_vec())
                .collect::<Vec<_>>(),
            self.t5_model.device(),
        )?;

        if !self.scheduler_config.use_dynamic_shifting {
            match t5_input_ids.dim(1)?.cmp(&256) {
                Ordering::Greater => {
                    candle_core::bail!("T5 embedding length greater than 256, please shrink the prompt or use the -dev (with guidance distillation) version.")
                }
                Ordering::Less | Ordering::Equal => {
                    t5_input_ids =
                        t5_input_ids.pad_with_zeros(D::Minus1, 0, 256 - t5_input_ids.dim(1)?)?;
                }
            }
        }

        let t5_embed = self.t5_model.forward(&t5_input_ids)?;

        let clip_input_ids = Tensor::new(
            self.clip_tokenizer
                .encode_batch(prompts, true)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                .into_iter()
                .map(|e| e.get_ids().to_vec())
                .collect::<Vec<_>>(),
            self.clip_model.device(),
        )?;
        let clip_embed = self.clip_model.forward(&clip_input_ids)?;

        let img: Tensor = sampling::get_noise(
            t5_embed.dim(0)?,
            params.height,
            params.width,
            t5_embed.device(),
        )?
        .to_dtype(t5_embed.dtype())?;

        let state = sampling::State::new(&t5_embed, &clip_embed, &img)?;
        let shift = if self.scheduler_config.use_dynamic_shifting {
            Some((
                state.img.dims()[1],
                self.scheduler_config.base_shift,
                self.scheduler_config.max_shift,
            ))
        } else {
            None
        };
        let timesteps = sampling::get_schedule(
            params
                .num_steps
                .unwrap_or(if self.scheduler_config.use_dynamic_shifting {
                    50
                } else {
                    4
                }),
            shift,
            self.scheduler_config.base_image_seq_len,
            self.scheduler_config.max_image_seq_len,
        );

        let img = if self.scheduler_config.use_dynamic_shifting {
            sampling::denoise(
                &self.flux_model,
                &state.img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &state.vec,
                &timesteps,
                params.guidance_scale,
            )?
        } else {
            sampling::denoise_no_guidance(
                &self.flux_model,
                &state.img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &state.vec,
                &timesteps,
            )?
        };

        let latent_img = sampling::unpack(&img, params.height, params.width)?;

        let latent_img =
            ((latent_img / self.vae_model.scale_factor())? + self.vae_model.shift_factor())?;
        let img = self.vae_model.decode(&latent_img)?;

        let normalized_img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;

        Ok(normalized_img)
    }
}
