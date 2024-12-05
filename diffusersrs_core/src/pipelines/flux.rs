use std::{collections::HashMap, fs, sync::Arc};

use anyhow::Result;
use candle_core::Device;
use serde::Deserialize;
use tokenizers::{models::bpe::BPE, ModelWrapper, Tokenizer};

use crate::{
    models::{
        dispatch_load_vae_model, ClipTextConfig, ClipTextTransformer, FluxConfig, FluxModel,
        T5Config, T5EncoderModel, VAEModel,
    },
    util::from_mmaped_safetensors,
};

use super::{ComponentElem, Loader, ModelPipeline};

pub struct FluxLoader;

#[derive(Clone, Debug, Deserialize)]
struct SchedulerConfig {
    base_image_seq_len: usize,
    base_shift: f64,
    max_image_seq_len: usize,
    max_shift: f64,
    num_train_timesteps: usize,
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
        dbg!(&flux_component);

        let pipeline = FluxPipeline {
            clip_tokenizer,
            clip_model: clip_component,
            t5_tokenizer,
            t5_model: t5_component,
            vae_model: vae_component,
            flux_model: flux_component,
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
}

impl ModelPipeline for FluxPipeline {}
