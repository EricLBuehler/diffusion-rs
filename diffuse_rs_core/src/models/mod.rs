mod clip;
mod flux;
mod t5;
mod vaes;

use std::sync::Arc;

pub use clip::{ClipTextConfig, ClipTextTransformer};
use diffuse_rs_backend::QuantMethod;
use diffuse_rs_common::core::Result;
pub use flux::{FluxConfig, FluxModel};
pub use t5::{T5Config, T5EncoderModel};

pub(crate) use vaes::{dispatch_load_vae_model, VAEModel};

pub struct QuantizedModelLayer<'a>(pub Vec<&'a mut Arc<dyn QuantMethod>>);

pub trait QuantizedModel {
    fn aggregate_layers(&mut self) -> Result<Vec<QuantizedModelLayer>>;
}
