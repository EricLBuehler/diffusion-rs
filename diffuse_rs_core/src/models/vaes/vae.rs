#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use diffuse_rs_common::core::{Result, Tensor, D};
use diffuse_rs_common::nn::{Activation, Conv2d, Conv2dConfig, GroupNorm};
use diffuse_rs_common::{conv2d, group_norm, linear, VarBuilder};
use serde::Deserialize;
use tracing::{span, Span};

fn default_act() -> Activation {
    Activation::Silu
}

#[derive(Debug, Clone, Deserialize)]
pub struct VAEConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    #[serde(default = "default_act")]
    pub act_fn: Activation,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub mid_block_add_attention: bool,
    pub down_block_types: Vec<String>,
    pub up_block_types: Vec<String>,
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    diffuse_rs_common::nn::ops::softmax_last_dim(&attn_weights)?.matmul(v)
}

#[derive(Debug, Clone)]
struct AttnBlock {
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    out: Conv2d,
    norm: GroupNorm,
    attn: Span,
}

impl AttnBlock {
    fn new(in_c: usize, vb: VarBuilder, cfg: &VAEConfig) -> Result<Self> {
        let q = linear(in_c, in_c, vb.pp("to_q"))?;
        let q = Conv2d::new(
            q.weight()
                .clone()
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?,
            q.bias().cloned(),
            Conv2dConfig::default(),
        );
        let k = linear(in_c, in_c, vb.pp("to_k"))?;
        let k = Conv2d::new(
            k.weight()
                .clone()
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?,
            k.bias().cloned(),
            Conv2dConfig::default(),
        );
        let v = linear(in_c, in_c, vb.pp("to_v"))?;
        let v = Conv2d::new(
            v.weight()
                .clone()
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?,
            v.bias().cloned(),
            Conv2dConfig::default(),
        );
        let out = linear(in_c, in_c, vb.pp("to_out.0"))?;
        let out = Conv2d::new(
            out.weight()
                .clone()
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?,
            out.bias().cloned(),
            Conv2dConfig::default(),
        );
        let norm = group_norm(cfg.norm_num_groups, in_c, 1e-6, vb.pp("group_norm"))?;
        Ok(Self {
            q,
            k,
            v,
            out,
            norm,
            attn: span!(tracing::Level::TRACE, "vae-attn"),
        })
    }
}

impl diffuse_rs_common::core::Module for AttnBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _span = self.attn.enter();
        let init_xs = xs;
        let xs = xs.apply(&self.norm)?;
        let q = xs.apply(&self.q)?;
        let k = xs.apply(&self.k)?;
        let v = xs.apply(&self.v)?;
        let (b, c, h, w) = q.dims4()?;
        let q = q.flatten_from(2)?.t()?.unsqueeze(1)?;
        let k = k.flatten_from(2)?.t()?.unsqueeze(1)?;
        let v = v.flatten_from(2)?.t()?.unsqueeze(1)?;
        let xs = scaled_dot_product_attention(&q, &k, &v)?;
        let xs = xs.squeeze(1)?.t()?.reshape((b, c, h, w))?;
        xs.apply(&self.out)? + init_xs
    }
}

#[derive(Debug, Clone)]
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    nin_shortcut: Option<Conv2d>,
    act_fn: Activation,
    resnet: Span,
}

impl ResnetBlock {
    fn new(in_c: usize, out_c: usize, vb: VarBuilder, cfg: &VAEConfig) -> Result<Self> {
        let conv_cfg = diffuse_rs_common::nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let norm1 = group_norm(cfg.norm_num_groups, in_c, 1e-6, vb.pp("norm1"))?;
        let conv1 = conv2d(in_c, out_c, 3, conv_cfg, vb.pp("conv1"))?;
        let norm2 = group_norm(cfg.norm_num_groups, out_c, 1e-6, vb.pp("norm2"))?;
        let conv2 = conv2d(out_c, out_c, 3, conv_cfg, vb.pp("conv2"))?;
        let nin_shortcut = if in_c == out_c {
            None
        } else {
            Some(conv2d(
                in_c,
                out_c,
                1,
                Default::default(),
                vb.pp("conv_shortcut"),
            )?)
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            nin_shortcut,
            act_fn: cfg.act_fn,
            resnet: span!(tracing::Level::TRACE, "vae-resnet"),
        })
    }
}

impl diffuse_rs_common::core::Module for ResnetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _span = self.resnet.enter();
        let h = xs
            .apply(&self.norm1)?
            .apply(&self.act_fn)?
            .apply(&self.conv1)?
            .apply(&self.norm2)?
            .apply(&self.act_fn)?
            .apply(&self.conv2)?;
        match self.nin_shortcut.as_ref() {
            None => xs + h,
            Some(c) => xs.apply(c)? + h,
        }
    }
}

#[derive(Debug, Clone)]
struct Downsample {
    conv: Conv2d,
    downsample: Span,
}

impl Downsample {
    fn new(in_c: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = diffuse_rs_common::nn::Conv2dConfig {
            stride: 2,
            ..Default::default()
        };
        let conv = conv2d(in_c, in_c, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self {
            conv,
            downsample: span!(tracing::Level::TRACE, "vae-downsample"),
        })
    }
}

impl diffuse_rs_common::core::Module for Downsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _span = self.downsample.enter();
        let xs = xs.pad_with_zeros(D::Minus1, 0, 1)?;
        let xs = xs.pad_with_zeros(D::Minus2, 0, 1)?;
        xs.apply(&self.conv)
    }
}

#[derive(Debug, Clone)]
struct Upsample {
    conv: Conv2d,
    upsample: Span,
}

impl Upsample {
    fn new(in_c: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = diffuse_rs_common::nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(in_c, in_c, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self {
            conv,
            upsample: span!(tracing::Level::TRACE, "vae-upsample"),
        })
    }
}

impl diffuse_rs_common::core::Module for Upsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _ = self.upsample.enter();
        let (_, _, h, w) = xs.dims4()?;
        xs.upsample_nearest2d(h * 2, w * 2)?.apply(&self.conv)
    }
}

#[derive(Debug, Clone)]
struct DownBlock {
    block: Vec<ResnetBlock>,
    downsample: Option<Downsample>,
}

#[derive(Debug, Clone)]
pub struct Encoder {
    conv_in: Conv2d,
    mid_block_1: ResnetBlock,
    mid_attn_1: Option<AttnBlock>,
    mid_block_2: ResnetBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
    down: Vec<DownBlock>,
    act_fn: Activation,
}

impl Encoder {
    pub fn new(cfg: &VAEConfig, vb: VarBuilder) -> Result<Self> {
        if !cfg
            .down_block_types
            .iter()
            .all(|x| x == "DownEncoderBlock2D")
        {
            diffuse_rs_common::bail!("All down (encoder) block types must be `DownEncoderBlock2D`");
        }
        let conv_cfg = diffuse_rs_common::nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let base_ch = cfg.block_out_channels[0];
        let mut block_in = base_ch;
        let conv_in = conv2d(cfg.in_channels, block_in, 3, conv_cfg, vb.pp("conv_in"))?;

        let mut down = Vec::with_capacity(cfg.block_out_channels.len());
        let vb_d = vb.pp("down_blocks");
        for (i_level, out_channels) in cfg.block_out_channels.iter().enumerate() {
            let mut block = Vec::with_capacity(cfg.layers_per_block);
            let vb_d = vb_d.pp(i_level);
            let vb_resnets = vb_d.pp("resnets");
            block_in = if i_level == 0 {
                base_ch
            } else {
                cfg.block_out_channels[i_level - 1]
            };
            let block_out = *out_channels;
            for i_block in 0..cfg.layers_per_block {
                let b = ResnetBlock::new(block_in, block_out, vb_resnets.pp(i_block), cfg)?;
                block.push(b);
                block_in = block_out;
            }
            let downsample = if i_level != cfg.block_out_channels.len() - 1 {
                Some(Downsample::new(block_in, vb_d.pp("downsamplers.0"))?)
            } else {
                None
            };
            let block = DownBlock { block, downsample };
            down.push(block)
        }

        // TODO: this is technically not general enough. Should always start with 1 resnet, then unet num_layers (defaults to 1 so this is OK)
        // repeats of attention and resnet!
        // https://github.com/huggingface/diffuse_rs/blob/243d9a49864ebb4562de6304a5fb9b9ebb496c6e/src/diffuse_rs/models/unets/unet_2d_blocks.py#L644-L729
        // https://github.com/huggingface/diffuse_rs/blob/243d9a49864ebb4562de6304a5fb9b9ebb496c6e/src/diffuse_rs/models/unets/unet_2d_blocks.py#L625
        let mid_block_1 = ResnetBlock::new(block_in, block_in, vb.pp("mid_block.resnets.0"), cfg)?;
        let mid_attn_1 = if cfg.mid_block_add_attention {
            Some(AttnBlock::new(
                block_in,
                vb.pp("mid_block.attentions.0"),
                cfg,
            )?)
        } else {
            None
        };
        let mid_block_2 = ResnetBlock::new(block_in, block_in, vb.pp("mid_block.resnets.1"), cfg)?;
        let conv_out = conv2d(
            block_in,
            2 * cfg.latent_channels,
            3,
            conv_cfg,
            vb.pp("conv_out"),
        )?;
        let norm_out = group_norm(cfg.norm_num_groups, block_in, 1e-6, vb.pp("conv_norm_out"))?;
        Ok(Self {
            conv_in,
            mid_block_1,
            mid_attn_1,
            mid_block_2,
            norm_out,
            conv_out,
            down,
            act_fn: cfg.act_fn,
        })
    }
}

impl diffuse_rs_common::nn::Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.apply(&self.conv_in)?;
        for block in self.down.iter() {
            for b in block.block.iter() {
                h = h.apply(b)?
            }
            if let Some(ds) = block.downsample.as_ref() {
                h = h.apply(ds)?
            }
        }
        h = h.apply(&self.mid_block_1)?;
        if let Some(attn) = &self.mid_attn_1 {
            h = h.apply(attn)?;
        }
        h.apply(&self.mid_block_2)?
            .apply(&self.norm_out)?
            .apply(&self.act_fn)?
            .apply(&self.conv_out)
    }
}

#[derive(Debug, Clone)]
struct UpBlock {
    block: Vec<ResnetBlock>,
    upsample: Option<Upsample>,
}

#[derive(Debug, Clone)]
pub struct Decoder {
    conv_in: Conv2d,
    mid_block_1: ResnetBlock,
    mid_attn_1: Option<AttnBlock>,
    mid_block_2: ResnetBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
    up: Vec<UpBlock>,
    act_fn: Activation,
}

impl Decoder {
    pub fn new(cfg: &VAEConfig, vb: VarBuilder) -> Result<Self> {
        if !cfg.up_block_types.iter().all(|x| x == "UpDecoderBlock2D") {
            diffuse_rs_common::bail!("All up (decoder) block types must be `UpDecoderBlock2D`");
        }
        let conv_cfg = diffuse_rs_common::nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let base_ch = cfg.block_out_channels[0];
        let mut block_in = cfg.block_out_channels.last().copied().unwrap_or(base_ch);

        // TODO: this is technically not general enough. Should always start with 1 resnet, then unet num_layers (defaults to 1 so this is OK)
        // repeats of attention and resnet!
        // https://github.com/huggingface/diffuse_rs/blob/243d9a49864ebb4562de6304a5fb9b9ebb496c6e/src/diffuse_rs/models/unets/unet_2d_blocks.py#L644-L729
        // https://github.com/huggingface/diffuse_rs/blob/243d9a49864ebb4562de6304a5fb9b9ebb496c6e/src/diffuse_rs/models/unets/unet_2d_blocks.py#L625
        let mid_block_1 = ResnetBlock::new(block_in, block_in, vb.pp("mid_block.resnets.0"), cfg)?;
        let mid_attn_1 = if cfg.mid_block_add_attention {
            Some(AttnBlock::new(
                block_in,
                vb.pp("mid_block.attentions.0"),
                cfg,
            )?)
        } else {
            None
        };
        let mid_block_2 = ResnetBlock::new(block_in, block_in, vb.pp("mid_block.resnets.1"), cfg)?;

        let conv_in = conv2d(cfg.latent_channels, block_in, 3, conv_cfg, vb.pp("conv_in"))?;

        let mut up = Vec::with_capacity(cfg.block_out_channels.len());
        let vb_u = vb.pp("up_blocks");
        for (i_level, out_channels) in cfg.block_out_channels.iter().rev().enumerate() {
            let block_out = *out_channels;
            let vb_u = vb_u.pp(i_level);
            let vb_resnets = vb_u.pp("resnets");
            let mut block = Vec::with_capacity(cfg.layers_per_block + 1);
            for i_block in 0..=cfg.layers_per_block {
                let b = ResnetBlock::new(block_in, block_out, vb_resnets.pp(i_block), cfg)?;
                block.push(b);
                block_in = block_out;
            }
            let upsample = if i_level != 3 {
                Some(Upsample::new(block_in, vb_u.pp("upsamplers.0"))?)
            } else {
                None
            };
            let block = UpBlock { block, upsample };
            up.push(block)
        }

        let norm_out = group_norm(cfg.norm_num_groups, base_ch, 1e-6, vb.pp("conv_norm_out"))?;
        let conv_out = conv2d(base_ch, cfg.out_channels, 3, conv_cfg, vb.pp("conv_out"))?;
        Ok(Self {
            conv_in,
            mid_block_1,
            mid_attn_1,
            mid_block_2,
            norm_out,
            conv_out,
            up,
            act_fn: cfg.act_fn,
        })
    }
}

impl diffuse_rs_common::nn::Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs.apply(&self.conv_in)?;
        let mut h = h.apply(&self.mid_block_1)?;
        if let Some(attn) = &self.mid_attn_1 {
            h = h.apply(attn)?;
        }
        h = h.apply(&self.mid_block_2)?;
        for block in self.up.iter() {
            for b in block.block.iter() {
                h = h.apply(b)?
            }
            if let Some(us) = block.upsample.as_ref() {
                h = h.apply(us)?
            }
        }
        h.apply(&self.norm_out)?
            .apply(&self.act_fn)?
            .apply(&self.conv_out)
    }
}

#[derive(Debug, Clone)]
pub struct DiagonalGaussian {
    sample: bool,
    chunk_dim: usize,
}

impl DiagonalGaussian {
    pub fn new(sample: bool, chunk_dim: usize) -> Result<Self> {
        Ok(Self { sample, chunk_dim })
    }
}

impl diffuse_rs_common::nn::Module for DiagonalGaussian {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let chunks = xs.chunk(2, self.chunk_dim)?;
        if self.sample {
            let std = (&chunks[1] * 0.5)?.exp()?;
            &chunks[0] + (std * chunks[0].randn_like(0., 1.))?
        } else {
            Ok(chunks[0].clone())
        }
    }
}
