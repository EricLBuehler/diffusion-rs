use candle_core::Result;
use candle_nn::{
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Embedding, GroupNorm, LayerNorm, LayerNormConfig,
    Linear,
};

use crate::VarBuilder;

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(ws, None))
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    let bs = vb.get(out_dim, "bias")?;
    Ok(Linear::new(ws, Some(bs)))
}

pub fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

pub fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv1d> {
    let ws = vb.get(
        (out_channels, in_channels / cfg.groups, kernel_size),
        "weight",
    )?;
    let bs = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(ws, Some(bs), cfg))
}

pub fn conv1d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv1d> {
    let ws = vb.get(
        (out_channels, in_channels / cfg.groups, kernel_size),
        "weight",
    )?;
    Ok(Conv1d::new(ws, None, cfg))
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv2d> {
    let ws = vb.get(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
    )?;
    let bs = vb.get(out_channels, "bias")?;
    Ok(Conv2d::new(ws, Some(bs), cfg))
}

pub fn conv2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv2d> {
    let ws = vb.get(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
    )?;
    Ok(Conv2d::new(ws, None, cfg))
}

pub fn group_norm(
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    vb: crate::VarBuilder,
) -> Result<GroupNorm> {
    let weight = vb.get(num_channels, "weight")?;
    let bias = vb.get(num_channels, "bias")?;
    GroupNorm::new(weight, bias, num_channels, num_groups, eps)
}

pub fn embedding(in_size: usize, out_size: usize, vb: crate::VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((in_size, out_size), "weight")?;
    Ok(Embedding::new(embeddings, out_size))
}

pub fn layer_norm<C: Into<LayerNormConfig>>(
    size: usize,
    config: C,
    vb: crate::VarBuilder,
) -> Result<LayerNorm> {
    let config: LayerNormConfig = config.into();
    assert!(config.remove_mean, "expected layernorm layer");

    let weight = vb.get(size, "weight")?;
    if config.affine {
        let bias = vb.get(size, "bias")?;
        Ok(LayerNorm::new(weight, bias, config.eps))
    } else {
        Ok(LayerNorm::new_no_bias(weight, config.eps))
    }
}
