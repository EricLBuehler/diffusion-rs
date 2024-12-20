use std::sync::Arc;

use autoencoder_kl::{AutencoderKlConfig, AutoEncoderKl};
use diffuse_rs_common::core::{Device, Result, Tensor};
use serde::Deserialize;

use diffuse_rs_common::{from_mmaped_safetensors, FileData, VarBuilder};

mod autoencoder_kl;
mod vae;

pub(crate) trait VAEModel: Send + Sync {
    #[allow(dead_code)]
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

fn load_autoencoder_kl(cfg_json: &FileData, vb: VarBuilder) -> anyhow::Result<Arc<dyn VAEModel>> {
    let cfg: AutencoderKlConfig = serde_json::from_str(&cfg_json.read_to_string()?)?;
    Ok(Arc::new(AutoEncoderKl::new(&cfg, vb)?))
}

pub(crate) fn dispatch_load_vae_model(
    cfg_json: &FileData,
    safetensor_files: Vec<FileData>,
    device: &Device,
) -> anyhow::Result<Arc<dyn VAEModel>> {
    let vb = from_mmaped_safetensors(safetensor_files, None, device, false)?;

    let VaeConfigShim { name } = serde_json::from_str(&cfg_json.read_to_string()?)?;
    match name.as_str() {
        "AutoencoderKL" => load_autoencoder_kl(cfg_json, vb),
        other => anyhow::bail!("Unexpected VAE type `{other:?}`."),
    }
}
