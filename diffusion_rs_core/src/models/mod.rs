mod clip;
mod flux;
mod t5;
mod vaes;

use std::sync::Arc;

pub use clip::{ClipTextConfig, ClipTextTransformer};
use diffusion_rs_backend::QuantMethod;
use diffusion_rs_common::core::{Device, Result};
pub use flux::{FluxConfig, FluxModel};
pub use t5::{T5Config, T5EncoderModel};

pub(crate) use vaes::{dispatch_load_vae_model, VAEModel};

#[derive(Debug)]
pub struct QuantizedModelLayer<'a>(pub Vec<&'a mut Arc<dyn QuantMethod>>);

pub trait QuantizedModel {
    /// Ensure that the devices of each layer match.
    fn match_devices_all_layers(&mut self, dev: &Device) -> Result<()>;
    /// Return all linear layers.
    fn aggregate_layers(&mut self) -> Result<Vec<QuantizedModelLayer>>;
    /// Cast all linear layers to the given device.
    fn to_device(&mut self, dev: &Device) -> Result<()> {
        let layers = self.aggregate_layers()?;
        for layer in layers {
            for x in layer.0 {
                *x = x.to_device(dev)?;
            }
        }
        self.match_devices_all_layers(dev)?;
        Ok(())
    }
    #[allow(unused)]
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
