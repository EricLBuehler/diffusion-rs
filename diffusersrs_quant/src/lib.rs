use std::{fmt::Debug, sync::Arc};

use bitsandbytes::Bnb4bitQuantLinear;
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use unquantized::UnquantLinear;

mod bitsandbytes;
mod unquantized;

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct QuantizedConfig {
    pub bnb_4bit_compute_dtype: String,
    pub bnb_4bit_quant_storage: String,
    pub bnb_4bit_quant_type: String,
    pub bnb_4bit_use_double_quant: bool,
    pub llm_int8_enable_fp32_cpu_offload: bool,
    pub llm_int8_has_fp16_weight: bool,
    pub llm_int8_skip_modules: bool,
    pub llm_int8_threshold: bool,
    pub load_in_4bit: bool,
    pub load_in_8bit: bool,
    pub block_size: Option<usize>,
    pub quant_method: QuantMethod,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default)]
pub enum QuantMethod {
    #[serde(rename = "unreachable")]
    Unreachable,
    #[default]
    #[serde(rename = "bitsandbytes")]
    Bitsandbytes,
}

pub trait QuantLinear: Debug {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

impl Module for dyn QuantLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Self::forward(self, xs)
    }
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    config: Option<&QuantizedConfig>,
    vb: VarBuilder,
) -> Result<Arc<dyn QuantLinear>> {
    let layer = if let Some(quant_conf) = &config {
        match quant_conf.quant_method {
            QuantMethod::Bitsandbytes => {
                Arc::new(Bnb4bitQuantLinear::linear_b(in_dim, out_dim, false, vb)?) as Arc<_>
            }
            QuantMethod::Unreachable => unreachable!(),
        }
    } else {
        let layer = candle_nn::linear_no_bias(in_dim, out_dim, vb)?;
        Arc::new(UnquantLinear(layer)) as Arc<_>
    };
    Ok(layer)
}

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    config: Option<&QuantizedConfig>,
    vb: VarBuilder,
) -> Result<Arc<dyn QuantLinear>> {
    let layer = if let Some(quant_conf) = &config {
        match quant_conf.quant_method {
            QuantMethod::Bitsandbytes => {
                Arc::new(Bnb4bitQuantLinear::linear_b(in_dim, out_dim, true, vb)?) as Arc<_>
            }
            QuantMethod::Unreachable => unreachable!(),
        }
    } else {
        let layer = candle_nn::linear(in_dim, out_dim, vb)?;
        Arc::new(UnquantLinear(layer)) as Arc<_>
    };
    Ok(layer)
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    config: Option<&QuantizedConfig>,
    vb: VarBuilder,
) -> Result<Arc<dyn QuantLinear>> {
    if bias {
        linear(in_dim, out_dim, config, vb)
    } else {
        linear_no_bias(in_dim, out_dim, config, vb)
    }
}
