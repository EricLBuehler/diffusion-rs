// https://github.com/huggingface/text-embeddings-inference/blob/cc1c510e8d8af8447c01e6b14c417473cf2dfda9/backends/candle/src/layers/cublaslt.rs

#![allow(unused_variables, unused_imports, dead_code)]

use diffusion_rs_common::core::{Device, Result, Tensor};
use diffusion_rs_common::nn::Activation as CandleActivation;
use once_cell::sync::Lazy;
use std::sync::{Mutex, Once};

#[cfg(feature = "cuda")]
mod api;
#[cfg(feature = "cuda")]
mod matmul;

#[cfg(feature = "cuda")]
pub use api::{fused_batch_matmul, CublasLt};

pub enum F8MatmulOutType {
    F8,
    BF16,
}

static INIT: Once = Once::new();
static mut CUBLASLT: Option<CublasLtWrapper> = None;
pub static CUBLASLT_HANDLE: Lazy<Mutex<Option<&'static CublasLtWrapper>>> =
    Lazy::new(|| Mutex::new(None));

pub fn maybe_init_cublas_lt_wrapper() {
    unsafe {
        INIT.call_once(|| {
            #[cfg(not(feature = "cuda"))]
            {
                CUBLASLT = None;
            }

            #[cfg(feature = "cuda")]
            {
                // Check if we can call the driver
                // Then check if we can create a device
                // Then check that the device is CUDA
                use diffusion_rs_common::core::cuda_backend::cudarc::driver;
                CUBLASLT = driver::result::init()
                    .ok()
                    .and_then(|_| Device::cuda_if_available(0).ok())
                    .and_then(|device| match device {
                        Device::Cuda(_) => Some(CublasLtWrapper {
                            cublaslt: CublasLt::new(&device).unwrap(),
                        }),
                        _ => None,
                    });
            }
            #[allow(static_mut_refs)]
            let cublaslt: Option<&'static CublasLtWrapper> = CUBLASLT.as_ref();
            *CUBLASLT_HANDLE.lock().unwrap() = cublaslt;
        });
    }
}

#[derive(Debug, Clone)]
pub struct CublasLtWrapper {
    #[cfg(feature = "cuda")]
    pub cublaslt: CublasLt,
}

impl CublasLtWrapper {
    /// Fused batch matmul + add + Relu/Gelu activation using CublasLt.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor of size BxMxK
    /// * `b` - Input tensor of size BxNxK
    /// * `out` - Optional Output tensor of size BxNxK.
    ///           If set and beta != 0, will be added to the end result of A*B before `act`
    /// * `alpha` - Optional scaling factor for A*B
    /// * `beta` - Optional scaling factor for C
    /// * `bias` - Optional bias tensor of size M
    /// * `act` - Optional Gelu or Relu activation. If set, will be added to the end result
    ///
    /// The resulting tensor is of shape NxM
    #[allow(clippy::too_many_arguments)]
    pub fn batch_matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: Option<&Tensor>,
        alpha: Option<f32>,
        beta: Option<f32>,
        bias: Option<&Tensor>,
        act: Option<CandleActivation>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let inner_act = act.map(|a| match a {
                CandleActivation::Relu => matmul::Activation::Relu,
                CandleActivation::Gelu => matmul::Activation::Gelu,
                _ => unreachable!("Unsupported activation in cublaslt matmul"),
            });
            let mut result = fused_batch_matmul(
                a,
                b,
                out,
                alpha,
                beta,
                bias,
                inner_act,
                self.cublaslt.clone(),
            )?;

            if Some(CandleActivation::Swiglu) == act {
                result = diffusion_rs_common::nn::ops::swiglu(&result)?;
            }
            Ok(result)
        }
        #[cfg(not(feature = "cuda"))]
        {
            diffusion_rs_common::bail!("`cuda` feature is not enabled")
        }
    }
}
