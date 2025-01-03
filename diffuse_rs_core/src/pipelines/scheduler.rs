use diffuse_rs_common::core::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct SchedulerConfig {
    #[serde(rename = "_class_name")]
    pub scheduler_type: SchedulerType,
    pub base_image_seq_len: usize,
    pub base_shift: f64,
    pub max_image_seq_len: usize,
    pub max_shift: f64,
    pub shift: f64,
    pub use_dynamic_shifting: bool,
}

#[derive(Deserialize, Clone)]
pub enum SchedulerType {
    #[serde(rename = "FlowMatchEulerDiscreteScheduler")]
    FlowMatchEulerDiscrete,
}

fn time_shift(mu: f64, sigma: f64, t: f64) -> f64 {
    let e = mu.exp();
    e / (e + (1. / t - 1.).powf(sigma))
}

impl SchedulerConfig {
    pub fn get_timesteps(&self, num_steps: usize, mu: Option<f64>) -> Result<Vec<f64>> {
        let mut sigmas: Vec<f64> = (0..=num_steps)
            .map(|v| v as f64 / num_steps as f64)
            .rev()
            .collect();
        match self.scheduler_type {
            SchedulerType::FlowMatchEulerDiscrete => {
                if self.use_dynamic_shifting {
                    let mu = mu.context("`mu` is required for dynamic shifting")?;
                    sigmas = sigmas
                        .iter()
                        .map(|sigma| time_shift(mu, 1., *sigma))
                        .collect();
                } else {
                    sigmas = sigmas
                        .iter()
                        .map(|sigma| self.shift * sigma / (1. + (self.shift - 1.) * sigma))
                        .collect();
                }

                Ok(sigmas)
            }
        }
    }
}
