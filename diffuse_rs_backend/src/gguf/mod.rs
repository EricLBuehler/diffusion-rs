use std::sync::Arc;

use diffuse_rs_common::core::{quantized::QMatMul, DType, Result, Tensor};
use diffuse_rs_common::nn::Module;

use crate::{QuantMethod, QuantMethodConfig};

#[derive(Debug)]
pub struct GgufMatMul {
    pub(crate) w: QMatMul,
    pub(crate) b: Option<Tensor>,
}

impl QuantMethod for GgufMatMul {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { q_weight, b } => Ok(Self {
                w: QMatMul::from_arc(q_weight)?,
                b,
            }),
            QuantMethodConfig::Unquantized(_) | QuantMethodConfig::Bnb4bit { .. } => unreachable!(),
        }
    }

    fn dequantize_w(&self, out_ty: DType) -> Result<Tensor> {
        self.w.dequantize_f16()?.to_dtype(out_ty)
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        let x = self.w.forward(a)?;
        if let Some(ref b) = self.b {
            x.broadcast_add(&b.to_dtype(x.dtype())?)
        } else {
            Ok(x)
        }
    }

    fn forward_via_half(&self, a: &Tensor) -> Result<Tensor> {
        let x = self.w.forward_via_f16(a)?;
        if let Some(ref b) = self.b {
            x.broadcast_add(&b.to_dtype(x.dtype())?)
        } else {
            Ok(x)
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        Some(DType::F32)
    }

    fn maybe_to_gguf_quant(self: Arc<Self>) -> Result<Arc<dyn QuantMethod>> {
        Ok(self.clone())
    }
}
