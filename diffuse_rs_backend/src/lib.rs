use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use diffuse_rs_common::core::{
    quantized::{GgmlDType, QTensor},
    DType, Result, Tensor,
};

#[cfg(feature = "metal")]
mod metal_kernels;

mod bitsandbytes;
mod cublaslt;
mod gguf;
pub mod ops;
mod unquantized;

pub use bitsandbytes::{BnbLinear, BnbQuantParmas, BnbQuantType};
pub use gguf::GgufMatMul;
pub use unquantized::UnquantLinear;

use diffuse_rs_common::nn::{Linear, Module};
use diffuse_rs_common::VarBuilder;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub enum QuantMethodType {
    #[serde(rename = "unreachable")]
    Unreachable,
    #[default]
    #[serde(rename = "bitsandbytes")]
    Bitsandbytes,
}

impl Display for QuantMethodType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bitsandbytes => write!(f, "bnb"),
            Self::Unreachable => write!(f, "unreachable",),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct QuantizedConfig {
    // GPTQ
    pub bits: Option<usize>,
    pub group_size: Option<usize>,
    pub checkpoint_format: Option<String>,

    // BNB
    pub bnb_4bit_quant_type: Option<String>,

    pub quant_method: QuantMethodType,
}

impl QuantizedConfig {
    pub fn get_bits_name(&self, _vb: &VarBuilder) -> String {
        match self.bits {
            Some(bits) => format!("{bits} bits"),
            None => {
                // Assume bnb
                self.bnb_4bit_quant_type
                    .clone()
                    .unwrap_or("int8".to_string())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum QuantMethodConfig {
    Gguf {
        q_weight: Arc<QTensor>,
        b: Option<Tensor>,
    },
    Unquantized(Linear),
    Bnb4bit {
        weight: Tensor,
        bias: Option<Tensor>,
        params: BnbQuantParmas,
        quant_ty: BnbQuantType,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq)]
pub enum IsqType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    HQQ8,
    HQQ4,
    // HQQ3,
    // HQQ2,
    // HQQ1,
    F8E4M3,
}

impl TryFrom<IsqType> for GgmlDType {
    type Error = diffuse_rs_common::core::Error;

    fn try_from(value: IsqType) -> Result<Self> {
        let tp = match value {
            IsqType::Q2K => Self::Q2K,
            IsqType::Q3K => Self::Q3K,
            IsqType::Q4K => Self::Q4K,
            IsqType::Q4_0 => Self::Q4_0,
            IsqType::Q4_1 => Self::Q4_1,
            IsqType::Q5K => Self::Q5K,
            IsqType::Q5_0 => Self::Q5_0,
            IsqType::Q5_1 => Self::Q5_1,
            IsqType::Q6K => Self::Q6K,
            IsqType::Q8K => Self::Q8K,
            IsqType::Q8_0 => Self::Q8_0,
            IsqType::Q8_1 => Self::Q8_1,
            _ => diffuse_rs_common::bail!("Expected valid GGML ISQ type."),
        };
        #[cfg(feature = "cuda")]
        {
            if !matches!(
                tp,
                GgmlDType::Q4_0
                    | GgmlDType::Q4_1
                    | GgmlDType::Q5_0
                    | GgmlDType::Q5_1
                    | GgmlDType::Q8_0
                    | GgmlDType::Q2K
                    | GgmlDType::Q3K
                    | GgmlDType::Q4K
                    | GgmlDType::Q5K
                    | GgmlDType::Q6K
            ) {
                diffuse_rs_common::bail!("GGML ISQ type on CUDA must be one of `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `HQQ8`, `HQQ4`")
            }
        }
        Ok(tp)
    }
}

/// Quantized method for a quantized matmul.
pub trait QuantMethod: Send + Sync + Debug {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized;

    fn dequantize_w(&self) -> Result<Tensor>;

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    /// Automatically cast to required quantization actiation type and back
    fn forward_autocast(&self, a: &Tensor) -> Result<Tensor> {
        let original_ty = a.dtype();
        let a = if let Some(t) = self.quantized_act_type() {
            a.to_dtype(t)?
        } else {
            a.clone()
        };
        self.forward(&a)?.to_dtype(original_ty)
    }

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    fn forward(&self, a: &Tensor) -> Result<Tensor>;

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    /// This may go via half precision if it is supported.
    fn forward_via_half(&self, a: &Tensor) -> Result<Tensor> {
        self.forward(a)
    }

    /// If a quantized method, return the activation dtype.
    fn quantized_act_type(&self) -> Option<DType>;

    #[deprecated(note = "do not use")]
    /// Convert to an equivalent gguf quantization, if applicable.
    fn maybe_to_gguf_quant(self: Arc<Self>) -> Result<Arc<dyn QuantMethod>>;
}

impl Module for dyn QuantMethod {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Self::forward(self, xs)
    }
}

fn vb_contains_quant(vb: &VarBuilder) -> bool {
    vb.contains_tensor("weight.absmax") || vb.contains_tensor("SCB")
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    vb: VarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    if vb_contains_quant(&vb) {
        if let Some(quant_conf) = &config {
            let layer = match quant_conf.quant_method {
                QuantMethodType::Bitsandbytes => {
                    Arc::new(BnbLinear::linear_b(in_dim, out_dim, false, vb)?) as Arc<_>
                }
                QuantMethodType::Unreachable => unreachable!(),
            };
            return Ok(layer);
        }
    }

    let ws = vb.get((out_dim, in_dim), "weight")?;
    let layer = Linear::new(ws, None);

    let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(layer))?;
    let layer = Arc::new(layer) as Arc<dyn QuantMethod>;
    Ok(layer)
}

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    vb: VarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    if vb_contains_quant(&vb) {
        if let Some(quant_conf) = &config {
            let layer = match quant_conf.quant_method {
                QuantMethodType::Bitsandbytes => {
                    Arc::new(BnbLinear::linear_b(in_dim, out_dim, true, vb)?) as Arc<_>
                }
                QuantMethodType::Unreachable => unreachable!(),
            };
            return Ok(layer);
        }
    }

    let ws = vb.get((out_dim, in_dim), "weight")?;
    let bs = vb.get(out_dim, "bias")?;
    let layer = Linear::new(ws, Some(bs));

    let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(layer))?;
    let layer = Arc::new(layer) as Arc<dyn QuantMethod>;
    Ok(layer)
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    config: &Option<QuantizedConfig>,
    vb: VarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    if bias {
        linear(in_dim, out_dim, config, vb)
    } else {
        linear_no_bias(in_dim, out_dim, config, vb)
    }
}
