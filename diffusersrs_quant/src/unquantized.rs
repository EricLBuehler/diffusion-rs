use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

use crate::QuantLinear;

#[derive(Debug)]
pub struct UnquantLinear(pub(crate) Linear);

impl QuantLinear for UnquantLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}
