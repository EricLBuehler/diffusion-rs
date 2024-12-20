#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Device, Result, Tensor};
use diffusers_cuda_graph::{copy_inplace, Graph, GraphInput};

use crate::models::FluxModel;
use diffusers_common::NiceProgressBar;

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

fn time_shift(mu: f64, sigma: f64, t: f64) -> f64 {
    let e = mu.exp();
    e / (e + (1. / t - 1.).powf(sigma))
}

/// `shift` is a triple `(image_seq_len, base_shift, max_shift)`.
pub fn get_schedule(
    num_steps: usize,
    shift: Option<(usize, f64, f64)>,
    base_seq_len: usize,
    max_seq_len: usize,
) -> Vec<f64> {
    let timesteps: Vec<f64> = (0..=num_steps)
        .map(|v| v as f64 / num_steps as f64)
        .rev()
        .collect();
    match shift {
        None => timesteps,
        Some((image_seq_len, y1, y2)) => {
            let (x1, x2) = (base_seq_len as f64, max_seq_len as f64);
            let m = (y2 - y1) / (x2 - x1);
            let b = y1 - m * x1;
            let mu = m * image_seq_len as f64 + b;
            timesteps
                .into_iter()
                .map(|v| time_shift(mu, 1., v))
                .collect()
        }
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

struct ModelInputs {
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Option<Tensor>,
}

impl GraphInput for ModelInputs {
    fn load_inputs_inplace(&self, input: Self, device: &Device) -> candle_core::Result<()> {
        unsafe { copy_inplace(&input.img, &self.img, device)? };
        unsafe { copy_inplace(&input.img_ids, &self.img_ids, device)? };
        unsafe { copy_inplace(&input.txt, &self.txt, device)? };
        unsafe { copy_inplace(&input.txt_ids, &self.txt_ids, device)? };
        unsafe { copy_inplace(&input.timesteps, &self.timesteps, device)? };
        unsafe { copy_inplace(&input.y, &self.y, device)? };
        if let Some(guidance) = &self.guidance {
            unsafe { copy_inplace(input.guidance.as_ref().unwrap(), &guidance, device)? };
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn denoise_inner(
    model: &FluxModel,
    img: &Tensor,
    img_ids: &Tensor,
    txt: &Tensor,
    txt_ids: &Tensor,
    vec_: &Tensor,
    timesteps: &[f64],
    guidance: Option<f64>,
) -> Result<Tensor> {
    let b_sz = img.dim(0)?;
    let dev = img.device();
    let guidance = if let Some(guidance) = guidance {
        Some(Tensor::full(guidance as f32, b_sz, dev)?)
    } else {
        None
    };
    let mut img = img.clone();
    let mut graph = None;
    for (i, window) in NiceProgressBar::<_, 'g'>(timesteps.windows(2).enumerate(), "Denoise loop") {
        let (t_curr, t_prev) = match window {
            [a, b] => (a, b),
            _ => continue,
        };
        #[cfg(feature = "cuda")]
        {
            let t_vec = Tensor::full(*t_curr as f32, b_sz, dev)?;
            let device = img.device().clone();
            if i == 0 {
                let initial_inputs = ModelInputs {
                    img: img.clone(),
                    img_ids: img_ids.clone(),
                    txt: txt.clone(),
                    txt_ids: txt_ids.clone(),
                    timesteps: t_vec.clone(),
                    y: vec_.clone(),
                    guidance: guidance.clone(),
                };
                graph = Some(Graph::new(
                    |input| {
                        let pred = model.forward(
                            &input.img,
                            &input.img_ids,
                            &input.txt,
                            &input.txt_ids,
                            &input.timesteps,
                            &input.y,
                            input.guidance.as_ref(),
                        )?;
                        img = (&img + pred * (t_prev - t_curr))?;
                        Ok(())
                    },
                    &device,
                    initial_inputs,
                )?);
            } else if let Some(graph) = &graph {
                let t_vec = Tensor::full(*t_curr as f32, b_sz, dev)?;
                graph.replay(ModelInputs {
                    img: img.clone(),
                    img_ids: img_ids.clone(),
                    txt: txt.clone(),
                    txt_ids: txt_ids.clone(),
                    timesteps: t_vec.clone(),
                    y: vec_.clone(),
                    guidance: guidance.clone(),
                })?;
            } else {
                unreachable!()
            }
        }
    }
    Ok(img)
}

#[allow(clippy::too_many_arguments)]
pub fn denoise(
    model: &FluxModel,
    img: &Tensor,
    img_ids: &Tensor,
    txt: &Tensor,
    txt_ids: &Tensor,
    vec_: &Tensor,
    timesteps: &[f64],
    guidance: f64,
) -> Result<Tensor> {
    denoise_inner(
        model,
        img,
        img_ids,
        txt,
        txt_ids,
        vec_,
        timesteps,
        Some(guidance),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_no_guidance(
    model: &FluxModel,
    img: &Tensor,
    img_ids: &Tensor,
    txt: &Tensor,
    txt_ids: &Tensor,
    vec_: &Tensor,
    timesteps: &[f64],
) -> Result<Tensor> {
    denoise_inner(model, img, img_ids, txt, txt_ids, vec_, timesteps, None)
}