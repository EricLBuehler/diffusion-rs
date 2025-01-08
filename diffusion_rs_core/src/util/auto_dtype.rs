use std::fmt::Display;

use anyhow::Result;
use diffusion_rs_common::core::{DType, Device, Tensor};
use serde::Deserialize;
use tracing::info;

#[derive(Clone, Copy, Default, Debug, Deserialize, PartialEq, clap::ValueEnum)]
/// DType for the model.
///
/// Note: When using `Auto`, fallback pattern is: BF16 -> F16 -> F32
pub enum ModelDType {
    #[default]
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "bf16")]
    BF16,
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "f32")]
    F32,
}

impl Display for ModelDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::BF16 => write!(f, "bf16"),
            Self::F16 => write!(f, "f16"),
            Self::F32 => write!(f, "f32"),
        }
    }
}

/// Type which can be converted to a DType
pub trait TryIntoDType {
    fn try_into_dtype(&self, devices: &[&Device], silent: bool) -> Result<DType>;
}

impl TryIntoDType for DType {
    fn try_into_dtype(&self, _: &[&Device], silent: bool) -> Result<DType> {
        if !silent {
            info!("dtype selected is {self:?}.");
        }
        if !matches!(self, DType::BF16 | DType::F32 | DType::F64 | DType::F16) {
            anyhow::bail!("DType must be one of BF16, F16, F32, F64");
        }
        Ok(*self)
    }
}

#[cfg(feature = "cuda")]
fn get_dtypes(silent: bool) -> Vec<DType> {
    use std::process::Command;

    // >= is supported
    const MIN_BF16_CC: usize = 800;
    // >= is supported
    const MIN_F16_CC: usize = 530;

    let raw_out = Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv")
        .output()
        .expect("Failed to run `nvidia-smi` but CUDA is selected.")
        .stdout;
    let out = String::from_utf8(raw_out).expect("`nvidia-smi` did not return valid utf8");
    // This reduce-min will always return at least one value so unwrap is OK.
    let min_cc = out
        .split('\n')
        .skip(1)
        .filter(|cc| !cc.trim().is_empty())
        .map(|cc| cc.trim().parse::<f32>().unwrap())
        .reduce(|a, b| if a < b { a } else { b })
        .unwrap();
    if !silent {
        info!("detected minimum CUDA compute capability {min_cc}");
    }
    // 7.5 -> 750
    #[allow(clippy::cast_possible_truncation)]
    let min_cc = (min_cc * 100.) as usize;

    let mut dtypes = Vec::new();
    if min_cc >= MIN_BF16_CC {
        dtypes.push(DType::BF16);
    } else if !silent {
        info!("skipping BF16 because CC < 8.0");
    }
    if min_cc >= MIN_F16_CC {
        dtypes.push(DType::F16);
    } else if !silent {
        info!("skipping F16 because CC < 5.3");
    }
    dtypes
}

fn get_dtypes_non_cuda() -> Vec<DType> {
    vec![DType::BF16, DType::F16]
}

#[cfg(not(feature = "cuda"))]
fn get_dtypes(_silent: bool) -> Vec<DType> {
    get_dtypes_non_cuda()
}

fn determine_auto_dtype_all(
    devices: &[&Device],
    silent: bool,
) -> diffusion_rs_common::core::Result<DType> {
    let dev_dtypes = get_dtypes(silent);
    for dtype in get_dtypes_non_cuda()
        .iter()
        .filter(|x| dev_dtypes.contains(x))
    {
        let mut results = Vec::new();
        for device in devices {
            // Try a matmul
            let x = Tensor::zeros((2, 2), *dtype, device)?;
            results.push(x.matmul(&x));
        }
        if results.iter().all(|x| x.is_ok()) {
            return Ok(*dtype);
        } else {
            for result in results {
                match result {
                    Ok(_) => (),
                    Err(e) => match e {
                        // For CUDA
                        diffusion_rs_common::core::Error::UnsupportedDTypeForOp(_, _) => continue,
                        // Accelerate backend doesn't support f16/bf16
                        // Metal backend doesn't support f16
                        diffusion_rs_common::core::Error::Msg(_) => continue,
                        // This is when the metal backend doesn't support bf16
                        diffusion_rs_common::core::Error::Metal(_) => continue,
                        // If running with RUST_BACKTRACE=1
                        diffusion_rs_common::core::Error::WithBacktrace { .. } => continue,
                        other => return Err(other),
                    },
                }
            }
        }
    }
    Ok(DType::F32)
}

impl TryIntoDType for ModelDType {
    fn try_into_dtype(&self, devices: &[&Device], silent: bool) -> Result<DType> {
        let dtype = match self {
            Self::Auto => {
                Ok(determine_auto_dtype_all(devices, silent).map_err(anyhow::Error::msg)?)
            }
            Self::BF16 => Ok(DType::BF16),
            Self::F16 => Ok(DType::F16),
            Self::F32 => Ok(DType::F32),
        };
        if !silent {
            info!("dtype selected is {:?}.", dtype.as_ref().unwrap());
        }
        dtype
    }
}
