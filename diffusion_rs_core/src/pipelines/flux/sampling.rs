#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use diffusion_rs_common::core::{Device, Result, Tensor};

pub fn get_noise(
    num_samples: usize,
    height: usize,
    width: usize,
    device: &Device,
) -> Result<Tensor> {
    let height = (height + 15) / 16 * 2;
    let width = (width + 15) / 16 * 2;
    Tensor::randn(0f32, 1., (num_samples, 16, height, width), device)
}

#[derive(Debug, Clone)]
pub struct State {
    pub img: Tensor,
    pub img_ids: Tensor,
    pub txt: Tensor,
    pub txt_ids: Tensor,
    pub vec: Tensor,
}

impl State {
    pub fn new(t5_emb: &Tensor, clip_emb: &Tensor, img: &Tensor) -> Result<Self> {
        let dtype = img.dtype();
        let (bs, c, h, w) = img.dims4()?;
        let dev = img.device();
        let img = img.reshape((bs, c, h / 2, 2, w / 2, 2))?; // (b, c, h, ph, w, pw)
        let img = img.permute((0, 2, 4, 1, 3, 5))?; // (b, h, w, c, ph, pw)
        let img = img.reshape((bs, h / 2 * w / 2, c * 4))?;
        let img_ids = Tensor::stack(
            &[
                Tensor::full(0u32, (h / 2, w / 2), dev)?,
                Tensor::arange(0u32, h as u32 / 2, dev)?
                    .reshape(((), 1))?
                    .broadcast_as((h / 2, w / 2))?,
                Tensor::arange(0u32, w as u32 / 2, dev)?
                    .reshape((1, ()))?
                    .broadcast_as((h / 2, w / 2))?,
            ],
            2,
        )?
        .to_dtype(dtype)?;
        let img_ids = img_ids.reshape((1, h / 2 * w / 2, 3))?;
        let img_ids = img_ids.repeat((bs, 1, 1))?;
        let txt = t5_emb.repeat(bs)?;
        let txt_ids = Tensor::zeros((bs, txt.dim(1)?, 3), dtype, dev)?;
        let vec = clip_emb.repeat(bs)?;
        Ok(Self {
            img,
            img_ids,
            txt,
            txt_ids,
            vec,
        })
    }
}

pub fn unpack(xs: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let (b, _h_w, c_ph_pw) = xs.dims3()?;
    let height = (height + 15) / 16;
    let width = (width + 15) / 16;
    xs.reshape((b, height, width, c_ph_pw / 4, 2, 2))? // (b, h, w, c, ph, pw)
        .permute((0, 3, 1, 4, 2, 5))? // (b, c, h, ph, w, pw)
        .reshape((b, c_ph_pw / 4, height * 2, width * 2))
}

pub fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len) as f64;
    let b = base_shift - m * base_seq_len as f64;
    image_seq_len as f64 * m + b
}
