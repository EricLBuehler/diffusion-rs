//! Core crate for interacting with diffuse_rs.
//!
//! The API is intentionally straightforward but strives to provide strong flexibility.
//!
//! ```rust
//! use std::time::Instant;
//!
//! use diffuse_rs_core::{DiffusionGenerationParams, ModelSource, Offloading, Pipeline, TokenSource};
//! use tracing::level_filters::LevelFilter;
//! use tracing_subscriber::EnvFilter;
//!
//! let filter = EnvFilter::builder()
//!     .with_default_directive(LevelFilter::INFO.into())
//!     .from_env_lossy();
//! tracing_subscriber::fmt().with_env_filter(filter).init();
//!
//! let pipeline = Pipeline::load(
//!     ModelSource::dduf("FLUX.1-dev-Q4-bnb.dduf")?,
//!     false,
//!     TokenSource::CacheToken,
//!     None,
//!     None,
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

pub use diffuse_rs_common::{ModelSource, TokenSource};
pub use pipelines::{DiffusionGenerationParams, Offloading, Pipeline};
