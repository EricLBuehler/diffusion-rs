//! Core crate for interacting with diffusion_rs.
//!
//! The API is intentionally straightforward but strives to provide strong flexibility.
//!
//! ```rust,no_run
//! use std::time::Instant;
//!
//! use diffusion_rs_core::{DiffusionGenerationParams, ModelSource, ModelDType, Offloading, Pipeline, TokenSource};
//!
//! let pipeline = Pipeline::load(
//!     ModelSource::dduf("FLUX.1-dev-Q4-bnb.dduf")?,
//!     true,
//!     TokenSource::CacheToken,
//!     None,
//!     None,
//!     &ModelDType::Auto,
//! )?;
//!
//! let start = Instant::now();
//!
//! let images = pipeline.forward(
//!     vec!["Draw a picture of a sunrise.".to_string()],
//!     DiffusionGenerationParams {
//!         height: 720,
//!         width: 1280,
//!         num_steps: 50,
//!         guidance_scale: 3.5,
//!     },
//! )?;
//!
//! let end = Instant::now();
//! println!("Took: {:.2}s", end.duration_since(start).as_secs_f32());
//!
//! images[0].save("image.png")?;
//!
//! # Ok::<(), anyhow::Error>(())
//! ```

mod models;
mod pipelines;
mod util;

pub use diffusion_rs_common::{ModelSource, TokenSource};
pub use pipelines::{DiffusionGenerationParams, Offloading, Pipeline};
pub use util::{ModelDType, TryIntoDType};
