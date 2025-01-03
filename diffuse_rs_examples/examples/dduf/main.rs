use std::time::Instant;

use clap::Parser;
use diffuse_rs_core::{DiffusionGenerationParams, ModelSource, Offloading, Pipeline, TokenSource};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
struct Args {
    /// DDUF file
    #[arg(long, short)]
    file: String,

    /// Prompt to use
    #[arg(short, long)]
    prompt: String,

    /// Guidance scale to use
    #[arg(short, long)]
    guidance_scale: f64,

    /// Number of denoising steps
    #[arg(short, long)]
    num_steps: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let pipeline = Pipeline::load(
        ModelSource::dduf(args.file)?,
        false,
        TokenSource::CacheToken,
        None,
        Offloading::Full,
    )?;

    let start = Instant::now();

    let images = pipeline.forward(
        vec![args.prompt],
        DiffusionGenerationParams {
            height: 720,
            width: 1280,
            num_steps: args.num_steps,
            guidance_scale: args.guidance_scale,
        },
    )?;

    let end = Instant::now();
    println!("Took: {:.2}s", end.duration_since(start).as_secs_f32());

    images[0].save("image.png")?;

    Ok(())
}
