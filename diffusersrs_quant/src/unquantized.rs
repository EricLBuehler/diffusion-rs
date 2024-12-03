use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

use crate::QuantLinear;

#[derive(Debug)]
pub struct UnquantLinear(pub(crate) Linear);

impl QuantLinear for UnquantLinear {
    fn dequantize_w(&self) -> Result<Tensor> {
        Ok(self.0.weight().clone())
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}
