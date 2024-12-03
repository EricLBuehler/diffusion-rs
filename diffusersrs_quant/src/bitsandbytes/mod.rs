use std::sync::Arc;

use candle_core::{Context, DType, Result, Shape, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::QuantLinear;

mod op;

const SUPPORTED_BLOCKSIZE: [usize; 7] = [2048, 4096, 1024, 512, 256, 128, 64];

#[derive(Debug, Deserialize, Clone, Copy)]
enum BnbDType {
    #[serde(rename = "float32")]
    F32,
    #[serde(rename = "bfloat16")]
    BF16,
    #[serde(rename = "float16")]
    F16,
}

#[derive(Debug, Clone, Copy)]
enum BnbQuantType {
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
struct BnbQuantState {
    blocksize: usize,
    shape: Vec<usize>,
    dtype: BnbDType,
    nested_blocksize: Option<usize>,
    nested_offset: Option<f64>,
    nested_dtype: Option<BnbDType>,
}

#[derive(Debug)]
struct BnbQuantParmas {
    absmax: Tensor,
    code: Tensor,
    blocksize: usize,
    shape: Option<Shape>,
    nested: Option<Arc<BnbQuantParmas>>,
    offset: Option<f64>,
    dtype: BnbDType,
}

#[derive(Debug)]
pub struct BnbQuantLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    params: BnbQuantParmas,
    quant_ty: BnbQuantType,
}

impl BnbQuantLinear {
    pub fn linear_b(_in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_unchecked_dtype("weight", DType::U8)?;

        let vb_w = vb.pp("weight");

        if !vb_w.contains_tensor("quant_state.bitsandbytes__nf4")
            && !vb_w.contains_tensor("quant_state.bitsandbytes__fp4")
        {
            candle_core::bail!("`BnbQuantLinear` expects either `...__nf4` or `...__fp4` tensors, this means the layer is not 4bit.");
        }

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

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
            candle_core::bail!("Only fp8/nf4 quantization is supported for now.")
        };

        let state_str = String::from_utf8(state.to_vec1::<u8>()?)?;
        let state: BnbQuantState =
            serde_json::from_str(&state_str).map_err(candle_core::Error::msg)?;

        let nested = if vb_w.contains_tensor("nested_absmax") {
            // TODO: can `nested_blocksize` be None, default to 64 like bnb?
            Some(Arc::new(BnbQuantParmas {
                absmax: vb_w.get_unchecked("nested_absmax")?,
                code: vb_w.get_unchecked("nested_quant_map")?,
                blocksize: state
                    .nested_blocksize
                    .context("`nested_blocksize` must be present.")?,
                shape: None,
                nested: None,
                offset: None, // Put it in the outer one!
                dtype: state
                    .nested_dtype
                    .context("`nested_dtype` must be present.")?,
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
            code: vb_w.get_unchecked("quant_map")?,
            blocksize: state.blocksize,
            shape: Some(Shape::from_dims(&state.shape)),
            nested,
            offset: state.nested_offset,
            dtype: state.dtype,
        };

        Ok(Self {
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
            absmax = (absmax + params.offset.context("`offset` must be present.")?)?;
        }

        let out_shape = params.shape.clone().unwrap_or(input.shape().clone());
        let out_dtype: DType = params.dtype.into();

        if !SUPPORTED_BLOCKSIZE.contains(&params.blocksize) {
            candle_core::bail!(
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

impl QuantLinear for BnbQuantLinear {
    fn dequantize_w(&self) -> Result<Tensor> {
        Self::dequantize(&self.weight, &self.params, self.quant_ty)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w = Self::dequantize(&self.weight, &self.params, self.quant_ty)?
            .t()?
            .to_dtype(xs.dtype())?;
        let res = xs.broadcast_matmul(&w)?;
        if let Some(bias) = &self.bias {
            res + bias
        } else {
            Ok(res)
        }
    }
}
