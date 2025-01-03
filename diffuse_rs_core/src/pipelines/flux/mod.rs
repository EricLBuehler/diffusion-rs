use std::sync::Mutex;
use std::{cmp::Ordering, collections::HashMap, sync::Arc};

use anyhow::Result;
use diffuse_rs_common::core::{DType, Device, Tensor, D};
use diffuse_rs_common::nn::Module;
use tokenizers::{models::bpe::BPE, ModelWrapper, Tokenizer};
use tracing::info;

use crate::models::QuantizedModel;
use crate::{
    models::{
        dispatch_load_vae_model, ClipTextConfig, ClipTextTransformer, FluxConfig, FluxModel,
        T5Config, T5EncoderModel, VAEModel,
    },
    pipelines::ComponentName,
};
use diffuse_rs_common::from_mmaped_safetensors;

use super::sampling::Sampler;
use super::scheduler::SchedulerConfig;
use super::{ComponentElem, DiffusionGenerationParams, Loader, ModelPipeline, Offloading};

mod sampling;

pub struct FluxLoader;

impl Loader for FluxLoader {
    fn name(&self) -> &'static str {
        "flux"
    }

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
        silent: bool,
        offloading_type: Offloading,
    ) -> Result<Arc<Mutex<dyn ModelPipeline>>> {
        let scheduler = components.remove(&ComponentName::Scheduler).unwrap();
        let clip_component = components.remove(&ComponentName::TextEncoder(1)).unwrap();
        let t5_component = components.remove(&ComponentName::TextEncoder(2)).unwrap();
        let clip_tok_component = components.remove(&ComponentName::Tokenizer(1)).unwrap();
        let t5_tok_component = components.remove(&ComponentName::Tokenizer(2)).unwrap();
        let flux_component = components.remove(&ComponentName::Transformer).unwrap();
        let vae_component = components.remove(&ComponentName::Vae).unwrap();

        let t5_flux_device = match offloading_type {
            Offloading::Full => Device::Cpu,
            Offloading::None => device.clone(),
        };

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
        if !silent {
            info!("loading CLIP model");
        }
        let clip_component = if let ComponentElem::Model {
            safetensors,
            config,
        } = clip_component
        {
            let cfg: ClipTextConfig = serde_json::from_str(&config.read_to_string()?)?;

            let vb =
                from_mmaped_safetensors(safetensors.into_values().collect(), None, device, silent)?;
            ClipTextTransformer::new(vb.pp("text_model"), &cfg)?
        } else {
            anyhow::bail!("incorrect storage of clip model")
        };
        if !silent {
            info!("loading T5 model");
        }
        let t5_component = if let ComponentElem::Model {
            safetensors,
            config,
        } = t5_component
        {
            let cfg: T5Config = serde_json::from_str(&config.read_to_string()?)?;
            let vb = from_mmaped_safetensors(
                safetensors.into_values().collect(),
                None,
                &t5_flux_device,
                silent,
            )?;
            T5EncoderModel::new(vb, &cfg)?
        } else {
            anyhow::bail!("incorrect storage of t5 model")
        };
        if !silent {
            info!("loading VAE model");
        }
        let vae_component = if let ComponentElem::Model {
            safetensors,
            config,
        } = vae_component
        {
            dispatch_load_vae_model(&config, safetensors.into_values().collect(), device, silent)?
        } else {
            anyhow::bail!("incorrect storage of vae model")
        };
        if !silent {
            info!("loading FLUX model");
        }
        let flux_component = if let ComponentElem::Model {
            safetensors,
            config,
        } = flux_component
        {
            let cfg: FluxConfig = serde_json::from_str(&config.read_to_string()?)?;
            let vb = from_mmaped_safetensors(
                safetensors.into_values().collect(),
                None,
                &t5_flux_device,
                silent,
            )?;
            FluxModel::new(&cfg, vb)?
        } else {
            anyhow::bail!("incorrect storage of flux model")
        };

        if !silent {
            info!(
                "FLUX pipeline using a guidance-distilled model: {}",
                flux_component.is_guidance()
            );
        }

        let pipeline = FluxPipeline {
            clip_tokenizer: Arc::new(clip_tokenizer),
            clip_model: clip_component,
            t5_tokenizer: Arc::new(t5_tokenizer),
            t5_model: t5_component,
            vae_model: vae_component,
            flux_model: flux_component,
            scheduler_config,
            device: device.clone(),
        };

        Ok(Arc::new(Mutex::new(pipeline)))
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
    device: Device,
}

impl FluxPipeline {
    fn tokenize_and_pad(
        prompts: Vec<String>,
        tokenizer: &Tokenizer,
    ) -> diffuse_rs_common::core::Result<Vec<Vec<u32>>> {
        let mut t5_tokens = Vec::new();
        let unpadded_t5_tokens = tokenizer
            .encode_batch(prompts, true)
            .map_err(|e| diffuse_rs_common::core::Error::Msg(e.to_string()))?
            .into_iter()
            .map(|e| e.get_ids().to_vec())
            .collect::<Vec<_>>();
        let t5_max_tokens = unpadded_t5_tokens.iter().map(|x| x.len()).max().unwrap();
        for mut tokenization in unpadded_t5_tokens {
            tokenization.extend(vec![0; t5_max_tokens - tokenization.len()]);
            t5_tokens.push(tokenization);
        }

        Ok(t5_tokens)
    }
}

impl ModelPipeline for FluxPipeline {
    fn forward(
        &mut self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
        offloading_type: Offloading,
    ) -> diffuse_rs_common::core::Result<Tensor> {
        match offloading_type {
            Offloading::Full => {
                self.t5_model.to_device(&self.device)?;
            }
            Offloading::None => (),
        }

        let mut t5_input_ids = Tensor::new(
            Self::tokenize_and_pad(prompts.clone(), &self.t5_tokenizer)?,
            &self.device,
        )?;

        if !self.flux_model.is_guidance() {
            match t5_input_ids.dim(1)?.cmp(&256) {
                Ordering::Greater => {
                    diffuse_rs_common::bail!("T5 embedding length greater than 256, please shrink the prompt or use the -dev (with guidance distillation) version.")
                }
                Ordering::Less | Ordering::Equal => {
                    t5_input_ids =
                        t5_input_ids.pad_with_zeros(D::Minus1, 0, 256 - t5_input_ids.dim(1)?)?;
                }
            }
        }

        let t5_embed = self.t5_model.forward(&t5_input_ids)?;

        match offloading_type {
            Offloading::Full => {
                self.t5_model.to_device(&Device::Cpu)?;
            }
            Offloading::None => (),
        }

        let clip_input_ids = Tensor::new(
            Self::tokenize_and_pad(prompts, &self.clip_tokenizer)?,
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
        let mu = sampling::calculate_shift(
            img.dims()[1],
            self.scheduler_config.base_image_seq_len,
            self.scheduler_config.max_image_seq_len,
            self.scheduler_config.base_shift,
            self.scheduler_config.max_shift,
        );
        let timesteps = self
            .scheduler_config
            .get_timesteps(params.num_steps, Some(mu))?;

        let bs = img.dim(0)?;
        let dev = img.device();

        match offloading_type {
            Offloading::Full => {
                self.flux_model.to_device(&self.device)?;
            }
            Offloading::None => (),
        }

        let guidance = if self.flux_model.is_guidance() {
            Some(Tensor::full(params.guidance_scale as f32, bs, dev)?)
        } else {
            None
        };
        let step = |img: &Tensor, t_vec: &Tensor| -> diffuse_rs_common::core::Result<Tensor> {
            self.flux_model.forward(
                img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                t_vec,
                &state.vec,
                guidance.as_ref(),
            )
        };

        let sampler = Sampler::new(&self.scheduler_config.scheduler_type);
        img = sampler.sample(&timesteps, &state.img, step)?;

        match offloading_type {
            Offloading::Full => {
                self.flux_model.to_device(&Device::Cpu)?;
            }
            Offloading::None => (),
        }

        img = sampling::unpack(&img, params.height, params.width)?;

        img = ((img / self.vae_model.scale_factor())? + self.vae_model.shift_factor())?;
        img = self.vae_model.decode(&img)?;

        img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;

        Ok(img)
    }
}
