use std::sync::Arc;

use diffusion_rs_common::core::Device;
use diffusion_rs_common::core::{quantized::QMatMul, DType, Result, Tensor};
use diffusion_rs_common::nn::Module;

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

    fn to_device(&self, dev: &Device) -> Result<Arc<dyn QuantMethod>> {
        let w = self.w.to_device(dev)?;
        let b = if let Some(b) = self.b.as_ref() {
            Some(b.to_device(dev)?)
        } else {
            None
        };
        Ok(Arc::new(Self { w, b }))
    }

    fn size_in_bytes(&self) -> Result<usize> {
        let w_size = self.w.size_in_bytes()?;
        let b_size = if let Some(b) = self.b.as_ref() {
            b.dtype().size_in_bytes() * b.elem_count()
        } else {
            0
        };
        Ok(w_size + b_size)
    }

    fn device(&self) -> Device {
        match &self.w {
            QMatMul::QTensor(q) => q.device(),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.device().clone(),
        }
    }
}
