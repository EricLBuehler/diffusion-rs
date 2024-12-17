use std::{cmp::Ordering, collections::HashMap, sync::Arc};

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
    pipelines::ComponentName,
};
use diffusers_common::from_mmaped_safetensors;

use super::{ComponentElem, DiffusionGenerationParams, Loader, ModelPipeline};

mod sampling;

pub struct FluxLoader;

#[derive(Clone, Debug, Deserialize)]
struct SchedulerConfig {
    base_image_seq_len: usize,
    base_shift: f64,
    max_image_seq_len: usize,
    max_shift: f64,
    // shift: f64,
    // use_dynamic_shifting: bool,
}

impl Loader for FluxLoader {
    fn required_component_names(&self) -> Vec<ComponentName> {
        vec![
            ComponentName::Scheduler,
            ComponentName::TextEncoder(1),
            ComponentName::TextEncoder(2),
            ComponentName::Tokenizer(1),
            ComponentName::Tokenizer(2),
            ComponentName::Transformer,
            ComponentName::Vae,
        ]
    }

    fn load_from_components(
        &self,
        mut components: HashMap<ComponentName, ComponentElem>,
        device: &Device,
    ) -> Result<Arc<dyn ModelPipeline>> {
        let scheduler = components.remove(&ComponentName::Scheduler).unwrap();
        let clip_component = components.remove(&ComponentName::TextEncoder(1)).unwrap();
        let t5_component = components.remove(&ComponentName::TextEncoder(2)).unwrap();
        let clip_tok_component = components.remove(&ComponentName::Tokenizer(1)).unwrap();
        let t5_tok_component = components.remove(&ComponentName::Tokenizer(2)).unwrap();
        let flux_component = components.remove(&ComponentName::Transformer).unwrap();
        let vae_component = components.remove(&ComponentName::Vae).unwrap();

        let scheduler_config = if let ComponentElem::Config { files } = scheduler {
            serde_json::from_str::<SchedulerConfig>(
                &files["scheduler/scheduler_config.json"].read_to_string()?,
            )?
        } else {
            anyhow::bail!("expected scheduler config")
        };
        let clip_tokenizer = if let ComponentElem::Other { files } = clip_tok_component {
            let vocab_file = &files["tokenizer/vocab.json"];
            let merges_file = &files["tokenizer/merges.txt"];
            let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_file.read_to_string()?)?;
            let merges: Vec<(String, String)> = merges_file
                .read_to_string()?
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
            Tokenizer::from_bytes(files["tokenizer_2/tokenizer.json"].read_to_string()?)
                .map_err(anyhow::Error::msg)?
        } else {
            anyhow::bail!("incorrect storage of t5 tokenizer")
        };
        let clip_component = if let ComponentElem::Model {
            safetensors,
            config,
        } = clip_component
        {
            let cfg: ClipTextConfig = serde_json::from_str(&config.read_to_string()?)?;

            let vb =
                from_mmaped_safetensors(safetensors.into_values().collect(), None, device, false)?;
            ClipTextTransformer::new(vb.pp("text_model"), &cfg)?
        } else {
            anyhow::bail!("incorrect storage of clip model")
        };
        let t5_component = if let ComponentElem::Model {
            safetensors,
            config,
        } = t5_component
        {
            let cfg: T5Config = serde_json::from_str(&config.read_to_string()?)?;
            let vb =
                from_mmaped_safetensors(safetensors.into_values().collect(), None, device, false)?;
            T5EncoderModel::new(vb, &cfg, device)?
        } else {
            anyhow::bail!("incorrect storage of t5 model")
        };
        let vae_component = if let ComponentElem::Model {
            safetensors,
            config,
        } = vae_component
        {
            dispatch_load_vae_model(&config, safetensors.into_values().collect(), device)?
        } else {
            anyhow::bail!("incorrect storage of vae model")
        };
        let flux_component = if let ComponentElem::Model {
            safetensors,
            config,
        } = flux_component
        {
            let cfg: FluxConfig = serde_json::from_str(&config.read_to_string()?)?;
            let vb =
                from_mmaped_safetensors(safetensors.into_values().collect(), None, device, false)?;
            FluxModel::new(&cfg, vb)?
        } else {
            anyhow::bail!("incorrect storage of flux model")
        };

        let pipeline = FluxPipeline {
            clip_tokenizer: Arc::new(clip_tokenizer),
            clip_model: clip_component,
            t5_tokenizer: Arc::new(t5_tokenizer),
            t5_model: t5_component,
            vae_model: vae_component,
            flux_model: flux_component,
            scheduler_config,
        };

        Ok(Arc::new(pipeline))
    }
}

pub struct FluxPipeline {
    clip_tokenizer: Arc<Tokenizer>,
    clip_model: ClipTextTransformer,
    t5_tokenizer: Arc<Tokenizer>,
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

        if !self.flux_model.is_guidance() {
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

        let mut img = sampling::get_noise(
            t5_embed.dim(0)?,
            params.height,
            params.width,
            t5_embed.device(),
        )?
        .to_dtype(t5_embed.dtype())?;

        let state = sampling::State::new(&t5_embed, &clip_embed, &img)?;
        let shift = if self.flux_model.is_guidance() {
            Some((
                state.img.dims()[1],
                self.scheduler_config.base_shift,
                self.scheduler_config.max_shift,
            ))
        } else {
            None
        };
        let timesteps = sampling::get_schedule(
            params.num_steps,
            shift,
            self.scheduler_config.base_image_seq_len,
            self.scheduler_config.max_image_seq_len,
        );

        img = if self.flux_model.is_guidance() {
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

        img = sampling::unpack(&img, params.height, params.width)?;

        img = ((img / self.vae_model.scale_factor())? + self.vae_model.shift_factor())?;
        img = self.vae_model.decode(&img)?;

        img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;

        Ok(img)
    }
}
