use diffuse_rs_common::{
    core::{Result, Tensor},
    NiceProgressBar,
};

use super::scheduler::SchedulerType;

pub enum Sampler {
    FlowMatchEulerDiscrete,
}

impl Sampler {
    pub fn new(ty: &SchedulerType) -> Self {
        match ty {
            SchedulerType::FlowMatchEulerDiscrete => Self::FlowMatchEulerDiscrete,
        }
    }

    /// Run the denoising process over the given image.
    ///
    /// Expects a step closure:
    /// ```ignore
    /// fn(img: &Tensor, t_vec: &Tensor) -> Result<Tensor>;
    /// ``````
    pub fn sample(
        &self,
        timesteps: &[f64],
        img: &Tensor,
        step: impl Fn(&Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        match self {
            Self::FlowMatchEulerDiscrete => {
                let b_sz = img.dim(0)?;
                let dev = img.device();
                let t_vec = Tensor::full(1f32, b_sz, dev)?;
                let mut img = img.clone();
                for window in NiceProgressBar::<_, 'g'>(timesteps.windows(2), "Denoise loop") {
                    let (t_curr, t_prev) = match window {
                        [a, b] => (a, b),
                        _ => continue,
                    };
                    let pred = step(&img, &(&t_vec * *t_curr)?)?;
                    img = (img + pred * (t_prev - t_curr))?
                }
                Ok(img)
            }
        }
    }
}
