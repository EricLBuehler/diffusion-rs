use std::{fs, path::PathBuf, sync::Arc};

use autoencoder_kl::{AutencoderKlConfig, AutoEncoderKl};
use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::util::from_mmaped_safetensors;

mod autoencoder_kl;
mod vae;

pub(crate) trait VAEModel {
    /// This function *does not* handle scaling the tensor! If you want to do this, apply the following to the output:
    /// `(x - vae.shift_factor())? * self.scale_factor()`
    fn encode(&self, xs: &Tensor) -> Result<Tensor>;

    /// This function *does not* handle scaling the tensor! If you want to do this, apply the following to the input:
    /// `(x / vae.scale_factor())? + self.shift_factor()`
    fn decode(&self, xs: &Tensor) -> Result<Tensor>;

    fn shift_factor(&self) -> f64;

    fn scale_factor(&self) -> f64;
}

#[derive(Clone, Debug, Deserialize)]
struct VaeConfigShim {
    #[serde(rename = "_class_name")]
    name: String,
}

fn load_autoencoder_kl(cfg_json: &PathBuf, vb: VarBuilder) -> anyhow::Result<Arc<dyn VAEModel>> {
    let cfg: AutencoderKlConfig = serde_json::from_str(&fs::read_to_string(cfg_json)?)?;
    Ok(Arc::new(AutoEncoderKl::new(&cfg, vb)?))
}

pub(crate) fn dispatch_load_vae_model(
    cfg_json: &PathBuf,
    safetensor_files: Vec<PathBuf>,
    device: &Device,
) -> anyhow::Result<Arc<dyn VAEModel>> {
    let vb = from_mmaped_safetensors(safetensor_files, None, device, false)?;

    let VaeConfigShim { name } = serde_json::from_str(&fs::read_to_string(cfg_json)?)?;
    match name.as_str() {
        "AutoencoderKL" => load_autoencoder_kl(cfg_json, vb),
        other => anyhow::bail!("Unexpected VAE type `{other:?}`."),
    }
}
