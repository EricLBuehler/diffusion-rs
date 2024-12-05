use candle_core::{Result, Tensor};
use candle_nn::{Activation, Conv2d, Conv2dConfig, VarBuilder};
use serde::Deserialize;

use super::{
    vae::{Decoder, DiagonalGaussian, Encoder, VAEConfig},
    VAEModel,
};

fn default_act() -> Activation {
    Activation::Silu
}

#[derive(Debug, Clone, Deserialize)]
pub struct AutencoderKlConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    #[serde(default = "default_act")]
    pub act_fn: Activation,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub scaling_factor: f64,
    pub shift_factor: f64,
    pub mid_block_add_attention: bool,
    pub use_quant_conv: bool,
    pub use_post_quant_conv: bool,
    pub down_block_types: Vec<String>,
    pub up_block_types: Vec<String>,
}

impl From<AutencoderKlConfig> for VAEConfig {
    fn from(value: AutencoderKlConfig) -> Self {
        Self {
            in_channels: value.in_channels,
            out_channels: value.out_channels,
            block_out_channels: value.block_out_channels,
            layers_per_block: value.layers_per_block,
            act_fn: value.act_fn,
            latent_channels: value.latent_channels,
            norm_num_groups: value.norm_num_groups,
            mid_block_add_attention: value.mid_block_add_attention,
            down_block_types: value.down_block_types,
            up_block_types: value.up_block_types,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AutoEncoderKl {
    encoder: Encoder,
    decoder: Decoder,
    reg: DiagonalGaussian,
    quant_conv: Option<Conv2d>,
    post_quant_conv: Option<Conv2d>,
    shift_factor: f64,
    scale_factor: f64,
}

impl AutoEncoderKl {
    pub fn new(cfg: &AutencoderKlConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(&cfg.clone().into(), vb.pp("encoder"))?;
        let decoder = Decoder::new(&cfg.clone().into(), vb.pp("decoder"))?;
        let reg = DiagonalGaussian::new(true, 1)?;
        let quant_conv = if cfg.use_quant_conv {
            Some(candle_nn::conv2d(
                2 * cfg.latent_channels,
                2 * cfg.latent_channels,
                1,
                Conv2dConfig::default(),
                vb.pp("quant_conv"),
            )?)
        } else {
            None
        };
        let post_quant_conv = if cfg.use_post_quant_conv {
            Some(candle_nn::conv2d(
                cfg.latent_channels,
                cfg.latent_channels,
                1,
                Conv2dConfig::default(),
                vb.pp("post_quant_conv"),
            )?)
        } else {
            None
        };
        Ok(Self {
            encoder,
            decoder,
            reg,
            scale_factor: cfg.scaling_factor,
            shift_factor: cfg.shift_factor,
            quant_conv,
            post_quant_conv,
        })
    }
}

impl VAEModel for AutoEncoderKl {
    fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let mut z = xs.apply(&self.encoder)?;
        if let Some(conv) = &self.quant_conv {
            z = z.apply(conv)?;
        }
        z = z.apply(&self.reg)?;
        // (z - self.shift_factor)? * self.scale_factor
        Ok(z)
    }

    fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        // let xs = ((xs / self.scale_factor)? + self.shift_factor)?;
        let mut z = xs.apply(&self.decoder)?;
        if let Some(conv) = &self.post_quant_conv {
            z = z.apply(conv)?;
        }
        Ok(z)
    }

    fn shift_factor(&self) -> f64 {
        self.shift_factor
    }

    fn scale_factor(&self) -> f64 {
        self.scale_factor
    }
}
