#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use diffusion_rs_backend::{QuantMethod, QuantizedConfig};
use diffusion_rs_common::core::{DType, Device, IndexOp, Result, Tensor, D};
use diffusion_rs_common::nn::{layer_norm::RmsNormNonQuantized, LayerNorm, RmsNorm};
use diffusion_rs_common::VarBuilder;
use serde::Deserialize;

use diffusion_rs_common::NiceProgressBar;
use tracing::{span, Span};

use crate::models::{QuantizedModel, QuantizedModelLayer};

const MLP_RATIO: f64 = 4.;
const HIDDEN_SIZE: usize = 3072;
const AXES_DIM: &[usize] = &[16, 56, 56];
const THETA: usize = 10000;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub in_channels: usize,
    pub pooled_projection_dim: usize,
    pub joint_attention_dim: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub guidance_embeds: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

fn layer_norm(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, vb.dtype(), vb.device())?;
    // Hack: use bias as 0s to take advantage of the fast kernel
    let bs = ws.zeros_like()?;
    Ok(LayerNorm::new(ws, bs, 1e-6))
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    diffusion_rs_backend::ops::sdpa(
        &q.to_dtype(DType::F32)?,
        &k.to_dtype(DType::F32)?,
        &v.to_dtype(DType::F32)?,
        scale_factor as f32,
        1.0,
    )?
    .to_dtype(q.dtype())

    // let mut batch_dims = q.dims().to_vec();
    // batch_dims.pop();
    // batch_dims.pop();
    // let q = q.flatten_to(batch_dims.len() - 1)?;
    // let k = k.flatten_to(batch_dims.len() - 1)?;
    // let v = v.flatten_to(batch_dims.len() - 1)?;
    // let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    // let attn_scores = diffusion_rs_common::nn::ops::softmax_last_dim(&attn_weights)?.matmul(&v)?;
    // batch_dims.push(attn_scores.dim(D::Minus2)?);
    // batch_dims.push(attn_scores.dim(D::Minus1)?);
    // attn_scores.reshape(batch_dims)
}

fn rope(pos: &Tensor, dim: usize, theta: usize) -> Result<Tensor> {
    if dim % 2 == 1 {
        diffusion_rs_common::bail!("dim {dim} is odd")
    }
    let dev = pos.device();
    let theta = theta as f64;
    let inv_freq: Vec<_> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32)
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, 1, inv_freq_len), dev)?;
    let inv_freq = inv_freq.to_dtype(pos.dtype())?;
    let freqs = pos.unsqueeze(2)?.broadcast_mul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    let out = Tensor::stack(&[&cos, &sin.neg()?, &sin, &cos], 3)?;
    let (b, n, d, _ij) = out.dims4()?;
    out.reshape((b, n, d, 2, 2))
}

fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
    let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
    (fr0.broadcast_mul(&x0)? + fr1.broadcast_mul(&x1)?)?.reshape(dims.to_vec())
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe: &Tensor) -> Result<Tensor> {
    let q = apply_rope(q, pe)?.contiguous()?;
    let k = apply_rope(k, pe)?.contiguous()?;
    let x = scaled_dot_product_attention(&q, &k, v)?;
    x.transpose(1, 2)?.flatten_from(2)
}

fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.;
    const MAX_PERIOD: f64 = 10000.;
    if dim % 2 == 1 {
        diffusion_rs_common::bail!("{dim} is odd")
    }
    let dev = t.device();
    let half = dim / 2;
    let t = (t * TIME_FACTOR)?;
    let arange =
        Tensor::arange(0, half as u32, dev)?.to_dtype(diffusion_rs_common::core::DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(diffusion_rs_common::core::DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)?;
    Ok(emb)
}

#[derive(Debug, Clone)]
pub struct EmbedNd {
    #[allow(unused)]
    dim: usize,
    theta: usize,
    axes_dim: Vec<usize>,
}

impl EmbedNd {
    fn new(dim: usize, theta: usize, axes_dim: Vec<usize>) -> Self {
        Self {
            dim,
            theta,
            axes_dim,
        }
    }
}

impl diffusion_rs_common::core::Module for EmbedNd {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let n_axes = ids.dim(D::Minus1)?;
        let mut emb = Vec::with_capacity(n_axes);
        for idx in 0..n_axes {
            let r = rope(
                &ids.get_on_dim(D::Minus1, idx)?,
                self.axes_dim[idx],
                self.theta,
            )?;
            emb.push(r)
        }
        let emb = Tensor::cat(&emb, 2)?;
        emb.unsqueeze(1)
    }
}

#[derive(Debug, Clone)]
pub struct MlpEmbedder {
    in_layer: Arc<dyn QuantMethod>,
    out_layer: Arc<dyn QuantMethod>,
}

impl MlpEmbedder {
    fn new(in_sz: usize, h_sz: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let in_layer =
            diffusion_rs_backend::linear(in_sz, h_sz, &cfg.quantization_config, vb.pp("linear_1"))?;
        let out_layer =
            diffusion_rs_backend::linear(h_sz, h_sz, &cfg.quantization_config, vb.pp("linear_2"))?;
        Ok(Self {
            in_layer,
            out_layer,
        })
    }
}

impl diffusion_rs_common::core::Module for MlpEmbedder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.out_layer
            .forward_autocast(&self.in_layer.forward_autocast(xs)?.silu()?)
    }
}

#[derive(Debug, Clone)]
pub struct QkNorm {
    query_norm: RmsNorm<RmsNormNonQuantized>,
    key_norm: RmsNorm<RmsNormNonQuantized>,
}

impl QkNorm {
    fn new(dim: usize, vb_q: VarBuilder, vb_k: VarBuilder) -> Result<Self> {
        let query_norm = vb_q.get(dim, "weight")?;
        let query_norm = RmsNorm::<RmsNormNonQuantized>::new(query_norm, 1e-6);
        let key_norm = vb_k.get(dim, "weight")?;
        let key_norm = RmsNorm::<RmsNormNonQuantized>::new(key_norm, 1e-6);
        Ok(Self {
            query_norm,
            key_norm,
        })
    }

    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            query_norm: self.query_norm.to_device(dev)?,
            key_norm: self.key_norm.to_device(dev)?,
        })
    }
}

struct ModulationOut {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl ModulationOut {
    fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.)?)?
            .broadcast_add(&self.shift)
    }

    fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

#[derive(Debug, Clone)]
struct Modulation1 {
    lin: Arc<dyn QuantMethod>,
    mod1: Span,
}

impl Modulation1 {
    fn new(dim: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let lin =
            diffusion_rs_backend::linear(dim, 3 * dim, &cfg.quantization_config, vb.pp("linear"))?;
        Ok(Self {
            lin,
            mod1: span!(tracing::Level::TRACE, "flux-mod1"),
        })
    }

    fn forward(&self, vec_: &Tensor) -> Result<ModulationOut> {
        let _span = self.mod1.enter();
        let ys = self
            .lin
            .forward_autocast(&vec_.silu()?)?
            .unsqueeze(1)?
            .chunk(3, D::Minus1)?;
        if ys.len() != 3 {
            diffusion_rs_common::bail!("unexpected len from chunk {ys:?}")
        }
        Ok(ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        })
    }
}

#[derive(Debug, Clone)]
struct Modulation2 {
    lin: Arc<dyn QuantMethod>,
    mod2: Span,
}

impl Modulation2 {
    fn new(dim: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let lin =
            diffusion_rs_backend::linear(dim, 6 * dim, &cfg.quantization_config, vb.pp("linear"))?;
        Ok(Self {
            lin,
            mod2: span!(tracing::Level::TRACE, "flux-mod2"),
        })
    }

    fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let _span = self.mod2.enter();
        let ys = self
            .lin
            .forward_autocast(&vec_.silu()?)?
            .unsqueeze(1)?
            .chunk(6, D::Minus1)?;
        if ys.len() != 6 {
            diffusion_rs_common::bail!("unexpected len from chunk {ys:?}")
        }
        let mod1 = ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        };
        let mod2 = ModulationOut {
            shift: ys[3].clone(),
            scale: ys[4].clone(),
            gate: ys[5].clone(),
        };
        Ok((mod1, mod2))
    }
}

#[derive(Debug, Clone)]
pub struct SelfAttention {
    q: Arc<dyn QuantMethod>,
    k: Arc<dyn QuantMethod>,
    v: Arc<dyn QuantMethod>,
    norm: QkNorm,
    proj: Arc<dyn QuantMethod>,
    num_attention_heads: usize,
    qkv: Span,
    fwd: Span,
}

impl SelfAttention {
    fn new(
        dim: usize,
        num_attention_heads: usize,
        qkv_bias: bool,
        cfg: &Config,
        vb: VarBuilder,
        context: bool,
    ) -> Result<Self> {
        let head_dim = dim / num_attention_heads;
        let (q, k, v, norm, proj) = if !context {
            let q = diffusion_rs_backend::linear_b(
                dim,
                dim,
                qkv_bias,
                &cfg.quantization_config,
                vb.pp("to_q"),
            )?;
            let k = diffusion_rs_backend::linear_b(
                dim,
                dim,
                qkv_bias,
                &cfg.quantization_config,
                vb.pp("to_k"),
            )?;
            let v = diffusion_rs_backend::linear_b(
                dim,
                dim,
                qkv_bias,
                &cfg.quantization_config,
                vb.pp("to_v"),
            )?;
            let norm = QkNorm::new(head_dim, vb.pp("norm_q"), vb.pp("norm_k"))?;
            let proj = diffusion_rs_backend::linear(
                dim,
                dim,
                &cfg.quantization_config,
                vb.pp("to_out.0"),
            )?;

            (q, k, v, norm, proj)
        } else {
            let q = diffusion_rs_backend::linear_b(
                dim,
                dim,
                qkv_bias,
                &cfg.quantization_config,
                vb.pp("add_q_proj"),
            )?;
            let k = diffusion_rs_backend::linear_b(
                dim,
                dim,
                qkv_bias,
                &cfg.quantization_config,
                vb.pp("add_k_proj"),
            )?;
            let v = diffusion_rs_backend::linear_b(
                dim,
                dim,
                qkv_bias,
                &cfg.quantization_config,
                vb.pp("add_v_proj"),
            )?;
            let norm = QkNorm::new(head_dim, vb.pp("norm_added_q"), vb.pp("norm_added_k"))?;
            let proj = diffusion_rs_backend::linear(
                dim,
                dim,
                &cfg.quantization_config,
                vb.pp("to_add_out"),
            )?;

            (q, k, v, norm, proj)
        };
        Ok(Self {
            q,
            k,
            v,
            norm,
            proj,
            num_attention_heads,
            qkv: span!(tracing::Level::TRACE, "flux-selfattn-qkv"),
            fwd: span!(tracing::Level::TRACE, "flux-selfattn-fwd"),
        })
    }

    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let _span = self.qkv.enter();
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.q.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut q = self.q.forward(&xs)?;
        let mut k = self.k.forward(&xs)?;
        let mut v = self.v.forward(&xs)?;
        if self.q.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }
        let (b, l, _khd) = q.dims3()?;
        q = q
            .reshape((b, l, self.num_attention_heads, ()))?
            .transpose(1, 2)?;
        k = k
            .reshape((b, l, self.num_attention_heads, ()))?
            .transpose(1, 2)?;
        v = v
            .reshape((b, l, self.num_attention_heads, ()))?
            .transpose(1, 2)?;
        q = q.apply(&self.norm.query_norm)?;
        k = k.apply(&self.norm.key_norm)?;
        Ok((q, k, v))
    }

    #[allow(unused)]
    fn forward(&self, xs: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let _span = self.fwd.enter();
        let (q, k, v) = self.qkv(xs)?;
        self.proj.forward_autocast(&attention(&q, &k, &v, pe)?)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    lin1: Arc<dyn QuantMethod>,
    lin2: Arc<dyn QuantMethod>,
    mlp: Span,
}

impl Mlp {
    fn new(in_sz: usize, mlp_sz: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let lin1 =
            diffusion_rs_backend::linear(in_sz, mlp_sz, &cfg.quantization_config, vb.pp("0.proj"))?;
        let lin2 =
            diffusion_rs_backend::linear(mlp_sz, in_sz, &cfg.quantization_config, vb.pp("2"))?;
        Ok(Self {
            lin1,
            lin2,
            mlp: span!(tracing::Level::TRACE, "flux-mlp"),
        })
    }
}

impl diffusion_rs_common::core::Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _span = self.mlp.enter();
        self.lin2
            .forward_autocast(&self.lin1.forward_autocast(xs)?.gelu()?)
    }
}

#[derive(Debug, Clone)]
pub struct DoubleStreamBlock {
    img_mod: Modulation2,
    img_norm1: LayerNorm,
    img_attn: SelfAttention,
    img_norm2: LayerNorm,
    img_mlp: Mlp,
    txt_mod: Modulation2,
    txt_norm1: LayerNorm,
    txt_attn: SelfAttention,
    txt_norm2: LayerNorm,
    txt_mlp: Mlp,
}

impl DoubleStreamBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h_sz = HIDDEN_SIZE;
        let mlp_sz = (h_sz as f64 * MLP_RATIO) as usize;
        let img_mod = Modulation2::new(h_sz, cfg, vb.pp("norm1"))?;
        let img_norm1 = layer_norm(h_sz, vb.pp("img_norm1"))?;
        let img_attn = SelfAttention::new(
            h_sz,
            cfg.num_attention_heads,
            true,
            cfg,
            vb.pp("attn"),
            false,
        )?;
        let img_norm2 = layer_norm(h_sz, vb.pp("img_norm2"))?;
        let img_mlp = Mlp::new(h_sz, mlp_sz, cfg, vb.pp("ff.net"))?;

        let txt_mod = Modulation2::new(h_sz, cfg, vb.pp("norm1_context"))?;
        let txt_norm1 = layer_norm(h_sz, vb.pp("txt_norm1"))?;
        let txt_attn = SelfAttention::new(
            h_sz,
            cfg.num_attention_heads,
            true,
            cfg,
            vb.pp("attn"),
            true,
        )?;
        let txt_norm2 = layer_norm(h_sz, vb.pp("txt_norm2"))?;
        let txt_mlp = Mlp::new(h_sz, mlp_sz, cfg, vb.pp("ff_context.net"))?;
        Ok(Self {
            img_mod,
            img_norm1,
            img_attn,
            img_norm2,
            img_mlp,
            txt_mod,
            txt_norm1,
            txt_attn,
            txt_norm2,
            txt_mlp,
        })
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (img_mod1, img_mod2) = self.img_mod.forward(vec_)?; // shift, scale, gate
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(vec_)?; // shift, scale, gate
        let img_modulated = img.apply(&self.img_norm1)?;
        let img_modulated = img_mod1.scale_shift(&img_modulated)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv(&img_modulated)?;

        let txt_modulated = txt.apply(&self.txt_norm1)?;
        let txt_modulated = txt_mod1.scale_shift(&txt_modulated)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv(&txt_modulated)?;

        let q = Tensor::cat(&[txt_q, img_q], 2)?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?;

        let attn = attention(&q, &k, &v, pe)?;
        let txt_attn = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;

        let img = (img + img_mod1.gate(&self.img_attn.proj.forward_autocast(&img_attn)?))?;
        let img = (&img
            + img_mod2.gate(
                &img_mod2
                    .scale_shift(&img.apply(&self.img_norm2)?)?
                    .apply(&self.img_mlp)?,
            )?)?;

        let txt = (txt + txt_mod1.gate(&self.txt_attn.proj.forward_autocast(&txt_attn)?))?;
        let txt = (&txt
            + txt_mod2.gate(
                &txt_mod2
                    .scale_shift(&txt.apply(&self.txt_norm2)?)?
                    .apply(&self.txt_mlp)?,
            )?)?;

        Ok((img, txt))
    }
}

#[derive(Debug, Clone)]
pub struct SingleStreamBlock {
    q: Arc<dyn QuantMethod>,
    k: Arc<dyn QuantMethod>,
    v: Arc<dyn QuantMethod>,
    proj_mlp: Arc<dyn QuantMethod>,
    linear2: Arc<dyn QuantMethod>,
    norm: QkNorm,
    pre_norm: LayerNorm,
    modulation: Modulation1,
    num_attention_heads: usize,
}

impl SingleStreamBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h_sz = HIDDEN_SIZE;
        let mlp_sz = (h_sz as f64 * MLP_RATIO) as usize;
        let head_dim = h_sz / cfg.num_attention_heads;

        let q = diffusion_rs_backend::linear_b(
            h_sz,
            h_sz,
            true,
            &cfg.quantization_config,
            vb.pp("attn.to_q"),
        )?;
        let k = diffusion_rs_backend::linear_b(
            h_sz,
            h_sz,
            true,
            &cfg.quantization_config,
            vb.pp("attn.to_k"),
        )?;
        let v = diffusion_rs_backend::linear_b(
            h_sz,
            h_sz,
            true,
            &cfg.quantization_config,
            vb.pp("attn.to_v"),
        )?;
        let proj_mlp = diffusion_rs_backend::linear_b(
            h_sz,
            mlp_sz,
            true,
            &cfg.quantization_config,
            vb.pp("proj_mlp"),
        )?;

        let linear2 = diffusion_rs_backend::linear(
            h_sz + mlp_sz,
            h_sz,
            &cfg.quantization_config,
            vb.pp("proj_out"),
        )?;
        let norm = QkNorm::new(head_dim, vb.pp("attn.norm_q"), vb.pp("attn.norm_k"))?;
        let pre_norm = layer_norm(h_sz, vb.pp("pre_norm"))?;
        let modulation = Modulation1::new(h_sz, cfg, vb.pp("norm"))?;
        Ok(Self {
            q,
            k,
            v,
            proj_mlp,
            linear2,
            norm,
            pre_norm,
            modulation,
            num_attention_heads: cfg.num_attention_heads,
        })
    }

    fn forward(&self, xs: &Tensor, vec_: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let mod_ = self.modulation.forward(vec_)?;
        let x_mod = mod_.scale_shift(&xs.apply(&self.pre_norm)?)?;
        let mut q = self.q.forward_autocast(&x_mod)?;
        let mut k = self.k.forward_autocast(&x_mod)?;
        let mut v = self.v.forward_autocast(&x_mod)?;
        let (b, l, _khd) = q.dims3()?;
        q = q
            .reshape((b, l, self.num_attention_heads, ()))?
            .transpose(1, 2)?;
        k = k
            .reshape((b, l, self.num_attention_heads, ()))?
            .transpose(1, 2)?;
        v = v
            .reshape((b, l, self.num_attention_heads, ()))?
            .transpose(1, 2)?;
        q = q.apply(&self.norm.query_norm)?;
        k = k.apply(&self.norm.key_norm)?;
        let mlp = self.proj_mlp.forward_autocast(&x_mod)?;
        let attn = attention(&q, &k, &v, pe)?;
        let output = self
            .linear2
            .forward_autocast(&Tensor::cat(&[attn, mlp.gelu()?], 2)?)?;
        xs + mod_.gate(&output)
    }
}

#[derive(Debug, Clone)]
pub struct LastLayer {
    norm_final: LayerNorm,
    linear: Arc<dyn QuantMethod>,
    ada_ln_modulation: Arc<dyn QuantMethod>,
}

impl LastLayer {
    fn new(h_sz: usize, p_sz: usize, out_c: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let norm_final = layer_norm(h_sz, vb.pp("norm_final"))?;
        let linear = diffusion_rs_backend::linear(
            h_sz,
            p_sz * p_sz * out_c,
            &cfg.quantization_config,
            vb.pp("proj_out"),
        )?;
        let ada_ln_modulation = diffusion_rs_backend::linear(
            h_sz,
            2 * h_sz,
            &cfg.quantization_config,
            vb.pp("norm_out.linear"),
        )?;
        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation,
        })
    }

    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = self
            .ada_ln_modulation
            .forward_autocast(&vec.silu()?)?
            .chunk(2, 1)?;
        let (scale, shift) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        self.linear.forward_autocast(&xs)
    }
}

#[derive(Debug, Clone)]
pub struct Flux {
    img_in: Arc<dyn QuantMethod>,
    txt_in: Arc<dyn QuantMethod>,
    time_in: MlpEmbedder,
    vector_in: MlpEmbedder,
    guidance_in: Option<MlpEmbedder>,
    pe_embedder: EmbedNd,
    double_blocks: Vec<DoubleStreamBlock>,
    single_blocks: Vec<SingleStreamBlock>,
    final_layer: LastLayer,
}

impl Flux {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let img_in = diffusion_rs_backend::linear(
            cfg.in_channels,
            HIDDEN_SIZE,
            &cfg.quantization_config,
            vb.pp("x_embedder"),
        )?;
        let txt_in = diffusion_rs_backend::linear(
            cfg.joint_attention_dim,
            HIDDEN_SIZE,
            &cfg.quantization_config,
            vb.pp("context_embedder"),
        )?;
        let mut double_blocks = Vec::with_capacity(cfg.num_layers);
        let vb_d = vb.pp("transformer_blocks");
        for idx in NiceProgressBar::<_, 'r'>(0..cfg.num_layers, "Loading double stream blocks") {
            let db = DoubleStreamBlock::new(cfg, vb_d.pp(idx))?;
            double_blocks.push(db)
        }
        let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
        let vb_s = vb.pp("single_transformer_blocks");
        for idx in
            NiceProgressBar::<_, 'r'>(0..cfg.num_single_layers, "Loading single stream blocks")
        {
            let sb = SingleStreamBlock::new(cfg, vb_s.pp(idx))?;
            single_blocks.push(sb)
        }
        let time_in = MlpEmbedder::new(
            256,
            HIDDEN_SIZE,
            cfg,
            vb.pp("time_text_embed.timestep_embedder"),
        )?;
        let vector_in = MlpEmbedder::new(
            cfg.pooled_projection_dim,
            HIDDEN_SIZE,
            cfg,
            vb.pp("time_text_embed.text_embedder"),
        )?;
        let guidance_in = if cfg.guidance_embeds {
            let mlp = MlpEmbedder::new(
                256,
                HIDDEN_SIZE,
                cfg,
                vb.pp("time_text_embed.guidance_embedder"),
            )?;
            Some(mlp)
        } else {
            None
        };
        let final_layer = LastLayer::new(HIDDEN_SIZE, 1, cfg.in_channels, cfg, vb)?;
        let pe_dim = HIDDEN_SIZE / cfg.num_attention_heads;
        let pe_embedder = EmbedNd::new(pe_dim, THETA, AXES_DIM.to_vec());

        Ok(Self {
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pe_embedder,
            double_blocks,
            single_blocks,
            final_layer,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        if txt.rank() != 3 {
            diffusion_rs_common::bail!("unexpected shape for txt {:?}", txt.shape())
        }
        if img.rank() != 3 {
            diffusion_rs_common::bail!("unexpected shape for img {:?}", img.shape())
        }
        let dtype = img.dtype();
        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?
        };
        let mut txt = self.txt_in.forward_autocast(txt)?;
        let mut img = self.img_in.forward_autocast(img)?;
        let vec_ = timestep_embedding(timesteps, 256, dtype)?.apply(&self.time_in)?;
        let vec_ = match (self.guidance_in.as_ref(), guidance) {
            (Some(g_in), Some(guidance)) => {
                (vec_ + timestep_embedding(guidance, 256, dtype)?.apply(g_in))?
            }
            _ => vec_,
        };
        let vec_ = (vec_ + y.apply(&self.vector_in))?;

        // Double blocks
        for block in self.double_blocks.iter() {
            (img, txt) = block.forward(&img, &txt, &vec_, &pe)?;
        }
        // Single blocks
        let mut img = Tensor::cat(&[&txt, &img], 1)?;
        for block in self.single_blocks.iter() {
            img = block.forward(&img, &vec_, &pe)?;
        }
        let img = img.i((.., txt.dim(1)?..))?;
        self.final_layer.forward(&img, &vec_)
    }

    pub fn is_guidance(&self) -> bool {
        self.guidance_in.is_some()
    }
}

impl QuantizedModel for Flux {
    fn match_devices_all_layers(&mut self, dev: &Device) -> Result<()> {
        self.final_layer = LastLayer {
            norm_final: self.final_layer.norm_final.to_device(dev)?,
            linear: self.final_layer.linear.clone(),
            ada_ln_modulation: self.final_layer.ada_ln_modulation.clone(),
        };

        for block in &mut self.double_blocks {
            block.img_attn.norm = block.img_attn.norm.to_device(dev)?;
            block.txt_attn.norm = block.txt_attn.norm.to_device(dev)?;

            block.img_norm1 = block.img_norm1.to_device(dev)?;
            block.img_norm2 = block.img_norm2.to_device(dev)?;
            block.txt_norm1 = block.txt_norm1.to_device(dev)?;
            block.txt_norm2 = block.txt_norm2.to_device(dev)?;
        }

        for block in &mut self.single_blocks {
            block.norm = block.norm.to_device(dev)?;

            block.pre_norm = block.pre_norm.to_device(dev)?;
        }
        Ok(())
    }

    fn aggregate_layers(&mut self) -> Result<Vec<QuantizedModelLayer>> {
        let mut layers = Vec::new();

        {
            let mut pre_layer_ct = vec![
                &mut self.txt_in,
                &mut self.img_in,
                &mut self.time_in.in_layer,
                &mut self.time_in.out_layer,
                &mut self.vector_in.in_layer,
                &mut self.vector_in.out_layer,
            ];

            if let Some(layer) = &mut self.guidance_in {
                pre_layer_ct.push(&mut layer.in_layer);
                pre_layer_ct.push(&mut layer.out_layer);
            }
            layers.push(QuantizedModelLayer(pre_layer_ct));
        }

        {
            let layer_ct = vec![
                &mut self.final_layer.ada_ln_modulation,
                &mut self.final_layer.linear,
            ];
            layers.push(QuantizedModelLayer(layer_ct));
        }

        for block in &mut self.double_blocks {
            let layer_ct = vec![
                &mut block.img_attn.q,
                &mut block.img_attn.k,
                &mut block.img_attn.v,
                &mut block.img_attn.proj,
                &mut block.img_mlp.lin1,
                &mut block.img_mlp.lin2,
                &mut block.img_mod.lin,
                &mut block.txt_attn.q,
                &mut block.txt_attn.k,
                &mut block.txt_attn.v,
                &mut block.txt_attn.proj,
                &mut block.txt_mlp.lin1,
                &mut block.txt_mlp.lin2,
                &mut block.txt_mod.lin,
            ];

            layers.push(QuantizedModelLayer(layer_ct));
        }

        for block in &mut self.single_blocks {
            let layer_ct = vec![
                &mut block.q,
                &mut block.k,
                &mut block.v,
                &mut block.modulation.lin,
                &mut block.proj_mlp,
                &mut block.linear2,
            ];

            layers.push(QuantizedModelLayer(layer_ct));
        }
        Ok(layers)
    }
}
