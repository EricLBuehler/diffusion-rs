use std::sync::Arc;

use candle_core::{Context, DType, Result, Shape, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::QuantLinear;

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
pub struct Bnb4bitQuantLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    params: BnbQuantParmas,
}

impl Bnb4bitQuantLinear {
    pub fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let vb_w = vb.pp("weight");

        if !vb_w.contains_tensor("quant_state.bitsandbytes__nf4")
            && !vb_w.contains_tensor("quant_state.bitsandbytes__fp4")
        {
            candle_core::bail!("`Bnb4bitQuantLinear` expects either `...__nf4` or `...__fp4` tensors, this means the layer is not 4bit.");
        }

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        let state = if vb_w.contains_tensor("quant_state.bitsandbytes__nf4") {
            vb_w.get_unchecked_dtype("quant_state.bitsandbytes__nf4", DType::U8)?
        } else if vb_w.contains_tensor("quant_state.bitsandbytes__fp4") {
            vb_w.get_unchecked_dtype("quant_state.bitsandbytes__fp4", DType::U8)?
        } else {
            unreachable!()
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

        let weight = vb.get_unchecked_dtype("weight", DType::U8)?;

        Ok(Self {
            weight,
            bias,
            params,
        })
    }

    /// Dequantize input (u8). Handles nested absmax dequantization.
    fn dequantize_blockwise(input: &Tensor, params: &BnbQuantParmas) -> Result<Tensor> {
        let mut absmax = params.absmax.clone();
        if let Some(nested) = &params.nested {
            absmax = Self::dequantize_blockwise(&params.absmax, &nested)?;
            absmax = (absmax + params.offset.context("`offset` must be present.")?)?;
        }

        let out = unsafe {
            Tensor::empty(
                params.shape.clone().unwrap_or(input.shape().clone()),
                params.dtype.into(),
                input.device(),
            )?
        };

        if !SUPPORTED_BLOCKSIZE.contains(&params.blocksize) {
            candle_core::bail!(
                "Blocksize of {} is not supported, {SUPPORTED_BLOCKSIZE:?} are.",
                params.blocksize
            );
        }

        todo!();
    }
}

impl QuantLinear for Bnb4bitQuantLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}
