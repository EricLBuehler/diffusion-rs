use std::sync::Arc;

use diffuse_rs_common::core::{
    quantized::{GgmlDType, QTensor},
    DType, Result, Shape, Tensor, D,
};
use diffuse_rs_common::VarBuilder;
use serde::Deserialize;

use crate::{GgufMatMul, QuantMethod, QuantMethodConfig};

#[cfg(feature = "cuda")]
mod ffi;

mod op;

const SUPPORTED_BLOCKSIZE: [usize; 7] = [2048, 4096, 1024, 512, 256, 128, 64];

#[derive(Debug, Deserialize, Clone, Copy)]
pub enum BnbDType {
    #[serde(rename = "float32")]
    F32,
    #[serde(rename = "bfloat16")]
    BF16,
    #[serde(rename = "float16")]
    F16,
}

#[derive(Debug, Clone, Copy)]
pub enum BnbQuantType {
    Int8,
    Fp4,
    Nf4,
}

impl From<BnbDType> for DType {
    fn from(value: BnbDType) -> Self {
        match value {
            BnbDType::F32 => Self::F32,
            BnbDType::BF16 => Self::BF16,
            BnbDType::F16 => Self::F16,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct BnbQuantState {
    pub blocksize: usize,
    pub shape: Vec<usize>,
    pub dtype: BnbDType,
    pub nested_blocksize: Option<usize>,
    pub nested_offset: Option<f64>,
    pub nested_dtype: Option<BnbDType>,
}

#[derive(Debug, Clone)]
pub struct BnbQuantParmas {
    pub absmax: Tensor,
    pub code: Tensor,
    pub blocksize: usize,
    pub shape: Option<Shape>,
    pub nested: Option<Arc<BnbQuantParmas>>,
    pub offset: Option<f64>,
    pub dtype: BnbDType,
}

#[derive(Debug)]
pub enum BnbLinear {
    Fp4Nf4 {
        weight: Tensor,
        bias: Option<Tensor>,
        params: BnbQuantParmas,
        quant_ty: BnbQuantType,
    },
    Int8 {
        weight: Tensor,
        scb: Tensor,
        bias: Option<Tensor>,
    },
}

impl BnbLinear {
    pub fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        if vb.contains_tensor("SCB") {
            Self::linear_8bit(in_dim, out_dim, bias, vb)
        } else if vb.contains_tensor("weight.quant_state.bitsandbytes__nf4")
            || vb.contains_tensor("weight.quant_state.bitsandbytes__fp4")
        {
            Self::linear_4bit(in_dim, out_dim, bias, vb)
        } else {
            diffuse_rs_common::bail!("`BnbLinear` expects fp4/nf4 or int8 layers.");
        }
    }

    fn linear_8bit(_in_dim: usize, _out_dim: usize, _bias: bool, _vb: VarBuilder) -> Result<Self> {
        // TODO: weight needs to be i8!

        // let weight = vb.get_unchecked_dtype("weight", DType::F32)?;
        // let scb = vb.get_unchecked_dtype("SCB", DType::F32)?;

        // let bias = if bias {
        //     Some(vb.get((out_dim,), "bias")?)
        // } else {
        //     None
        // };

        // Ok(Self::Int8 { weight, scb, bias })

        diffuse_rs_common::bail!("Int8 quantization is unsupported.");
    }

    fn linear_4bit(_in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_unchecked_dtype("weight", DType::U8)?;

        let vb_w = vb.pp("weight");

        if !vb_w.contains_tensor("quant_state.bitsandbytes__nf4")
            && !vb_w.contains_tensor("quant_state.bitsandbytes__fp4")
        {
            diffuse_rs_common::bail!("`BnbLinear` expects either `...__nf4` or `...__fp4` tensors, this means the layer is not 4bit or 8big.");
        }

        let quant_ty = if vb_w.contains_tensor("quant_state.bitsandbytes__nf4") {
            BnbQuantType::Nf4
        } else if vb_w.contains_tensor("quant_state.bitsandbytes__fp4") {
            BnbQuantType::Fp4
        } else {
            BnbQuantType::Int8
        };

        let state = match quant_ty {
            BnbQuantType::Nf4 => {
                Some(vb_w.get_unchecked_dtype("quant_state.bitsandbytes__nf4", DType::U8)?)
            }
            BnbQuantType::Fp4 => {
                Some(vb_w.get_unchecked_dtype("quant_state.bitsandbytes__fp4", DType::U8)?)
            }
            BnbQuantType::Int8 => None,
        };
        let Some(state) = state else {
            diffuse_rs_common::bail!("Only fp8/nf4 quantization is supported for now.")
        };

        let state_str = String::from_utf8(state.to_vec1::<u8>()?)?;
        let state: BnbQuantState =
            serde_json::from_str(&state_str).map_err(diffuse_rs_common::core::Error::msg)?;

        let nested = if vb_w.contains_tensor("nested_absmax") {
            // TODO: can `nested_blocksize` be None, default to 64 like bnb?
            Some(Arc::new(BnbQuantParmas {
                absmax: vb_w.get_unchecked_dtype("nested_absmax", DType::F32)?,
                code: vb_w.get_unchecked_dtype("nested_quant_map", DType::F32)?,
                blocksize: state
                    .nested_blocksize
                    .ok_or(diffuse_rs_common::core::Error::debug(
                        "`nested_blocksize` must be present.",
                    ))?,
                shape: None,
                nested: None,
                offset: None, // Put it in the outer one!
                dtype: state
                    .nested_dtype
                    .ok_or(diffuse_rs_common::core::Error::debug(
                        "`nested_dtype` must be present.",
                    ))?,
            }))
        } else {
            None
        };

        let absmax = if nested.is_some() {
            vb_w.get_unchecked_dtype("absmax", DType::U8)?
        } else {
            vb_w.get_unchecked_dtype("absmax", DType::F32)?
        };

        let params = BnbQuantParmas {
            absmax,
            code: vb_w.get_unchecked_dtype("quant_map", DType::F32)?,
            blocksize: state.blocksize,
            shape: Some(Shape::from_dims(&state.shape)),
            nested,
            offset: state.nested_offset,
            dtype: state.dtype,
        };

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?.to_dtype(params.dtype.into())?)
        } else {
            None
        };

        Ok(Self::Fp4Nf4 {
            weight,
            bias,
            params,
            quant_ty,
        })
    }

    /// Dequantize input (u8). Handles nested absmax dequantization.
    fn dequantize(
        input: &Tensor,
        params: &BnbQuantParmas,
        quant_ty: BnbQuantType,
    ) -> Result<Tensor> {
        let mut absmax = params.absmax.clone();
        if let Some(nested) = &params.nested {
            absmax = Self::dequantize(&params.absmax, nested, BnbQuantType::Int8)?;
            absmax = (absmax
                + params.offset.ok_or(diffuse_rs_common::core::Error::debug(
                    "`offset` must be present.",
                ))?)?;
        }

        let out_shape = params.shape.clone().unwrap_or(input.shape().clone());
        let out_dtype: DType = params.dtype.into();

        if !SUPPORTED_BLOCKSIZE.contains(&params.blocksize) {
            diffuse_rs_common::bail!(
                "Blocksize of {} is not supported, {SUPPORTED_BLOCKSIZE:?} are.",
                params.blocksize
            );
        }

        op::dequantize(
            input,
            &absmax,
            &params.code,
            out_shape,
            params.blocksize,
            quant_ty,
            params.dtype,
        )?
        .to_dtype(out_dtype)
    }
}

impl QuantMethod for BnbLinear {
    fn new(method: QuantMethodConfig) -> diffuse_rs_common::core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. } | QuantMethodConfig::Unquantized(_) => unreachable!(),
            QuantMethodConfig::Bnb4bit {
                weight,
                bias,
                params,
                quant_ty,
            } => Ok(Self::Fp4Nf4 {
                weight,
                bias,
                params,
                quant_ty,
            }),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        match self {
            Self::Fp4Nf4 {
                weight,
                bias: _,
                params,
                quant_ty,
            } => Self::dequantize(weight, params, *quant_ty),
            Self::Int8 {
                weight,
                scb,
                bias: _,
            } => {
                // https://huggingface.co/blog/hf-bitsandbytes-integration#hugging-face-transformers-integration-nuances
                weight
                    .to_dtype(scb.dtype())?
                    .broadcast_div(&scb.unsqueeze(1)?)?
                    / 127.
            }
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.dequantize_w()?.t()?.to_dtype(xs.dtype())?;
        let res = xs.broadcast_matmul(&w)?;
        let bias = match self {
            Self::Fp4Nf4 { bias, .. } | Self::Int8 { bias, .. } => bias,
        };
        if let Some(bias) = bias {
            res.broadcast_add(&bias.to_dtype(res.dtype())?)
        } else {
            Ok(res)
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn maybe_to_gguf_quant(self: Arc<Self>) -> Result<Arc<dyn QuantMethod>> {
        let weight = self.dequantize_w()?;

        match &*self {
            Self::Fp4Nf4 { bias, quant_ty, .. } => {
                let last_dim = weight.dim(D::Minus1)?;
                let dtype = match quant_ty {
                    BnbQuantType::Fp4 | BnbQuantType::Nf4 if last_dim % 256 == 0 => GgmlDType::Q4K,
                    BnbQuantType::Fp4 | BnbQuantType::Nf4
                        if last_dim % 64 == 0 && last_dim % 256 != 0 =>
                    {
                        GgmlDType::Q4_0
                    }
                    BnbQuantType::Fp4 | BnbQuantType::Nf4
                        if last_dim % 64 != 0 && last_dim % 256 != 0 =>
                    {
                        GgmlDType::F32
                    }
                    BnbQuantType::Int8 => GgmlDType::Q8_0,
                    _ => unreachable!(),
                };
                let qmatmul = QTensor::quantize(&weight, dtype)?;
                Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(qmatmul),
                    b: bias.clone(),
                })?))
            }
            Self::Int8 { bias, .. } => {
                let qmatmul = QTensor::quantize(&weight, GgmlDType::Q8_0)?;
                Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(qmatmul),
                    b: bias.clone(),
                })?))
            }
        }
    }
}
