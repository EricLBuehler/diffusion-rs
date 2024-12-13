use std::sync::Arc;

use candle_core::{DType, Device, DeviceLocation, Result, Shape, Tensor, D};

use crate::{
    cublaslt::{maybe_init_cublas_lt_wrapper, CUBLASLT_HANDLE},
    QuantMethod, QuantMethodConfig,
};

#[derive(Debug)]
pub struct UnquantLinear {
    w: Tensor,
    b: Option<Tensor>,
}

impl QuantMethod for UnquantLinear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. } | QuantMethodConfig::Bnb { .. } => unreachable!(),
            QuantMethodConfig::Unquantized(l) => Ok(Self {
                w: l.weight().clone(),
                b: l.bias().cloned(),
            }),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        Ok(self.w.clone())
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        // Batch matrix multiplication
        maybe_init_cublas_lt_wrapper();

        let w = match *a.dims() {
            [b1, b2, _, _] => self.w.broadcast_left((b1, b2))?,
            [bsize, _, _] => self.w.broadcast_left(bsize)?,
            _ => self.w.clone(),
        };

        if let Some(b) = self.b.as_ref() {
            let mut tgt_shape = a.dims().to_vec();
            tgt_shape[a.dims().len() - 1] = w.dim(D::Minus2)?;
            let b = b.broadcast_as(Shape::from_dims(&tgt_shape))?;

            match a.device().location() {
                DeviceLocation::Cuda { .. } => {
                    // Try to use cublaslt, otherwise fallback to gemm
                    if let (Device::Cuda(_), Some(cublaslt)) =
                        (a.device(), *CUBLASLT_HANDLE.lock().unwrap())
                    {
                        cublaslt
                            .batch_matmul(
                                a,
                                &w,
                                Some(&b.t()?.contiguous()?),
                                None,
                                Some(1.0),
                                None,
                                None,
                            )?
                            .t()
                    } else {
                        a.matmul(&w.t()?)?.broadcast_add(&b)
                    }
                }
                DeviceLocation::Metal { .. } | DeviceLocation::Cpu => {
                    a.matmul(&w.t()?)?.broadcast_add(&b)
                }
            }
        } else {
            a.matmul(&w.t()?)
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn maybe_to_gguf_quant(self: Arc<Self>) -> Result<Arc<dyn QuantMethod>> {
        Ok(self.clone())
    }
}
