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

#[derive(Debug)]
pub struct QuantizedModelLayer<'a>(pub Vec<&'a mut Arc<dyn QuantMethod>>);

pub trait QuantizedModel {
    fn aggregate_layers(&mut self) -> Result<Vec<QuantizedModelLayer>>;
    fn total_size_in_bytes(&mut self) -> Result<usize> {
        let layers = self.aggregate_layers()?;
        let mut total = 0;

        for layer in layers {
            for x in &layer.0 {
                total += x.size_in_bytes()?;
            }
        }

        Ok(total)
    }
}
