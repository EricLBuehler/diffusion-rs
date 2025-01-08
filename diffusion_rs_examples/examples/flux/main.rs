use std::time::Instant;

use diffusion_rs_core::{
    DiffusionGenerationParams, ModelDType, ModelSource, Offloading, Pipeline, TokenSource,
};

use clap::{Parser, ValueEnum};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    #[value(name = "schnell")]
    Schnell,
    #[value(name = "dev")]
    Dev,
}

#[derive(Parser)]
struct Args {
    /// Which model to use
    #[arg(long, default_value = "schnell")]
    which: Which,

    /// Prompt to use
    #[arg(short, long)]
    prompt: String,

    /// Offloading setting to use for this model
    #[arg(short, long)]
    offloading: Option<Offloading>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let model_id = match args.which {
        Which::Dev => "black-forest-labs/FLUX.1-dev",
        Which::Schnell => "black-forest-labs/FLUX.1-schnell",
    };

    let pipeline = Pipeline::load(
        ModelSource::from_model_id(model_id),
        false,
        TokenSource::CacheToken,
        None,
        args.offloading,
        &ModelDType::Auto,
    )?;
    let num_steps = match args.which {
        Which::Dev => 50,
        Which::Schnell => 4,
    };
    let guidance_scale = match args.which {
        Which::Dev => 3.5,
        Which::Schnell => 0.0,
    };

    let start = Instant::now();

    let images = pipeline.forward(
        vec![args.prompt],
        DiffusionGenerationParams {
            height: 720,
            width: 1280,
            num_steps,
            guidance_scale,
        },
    )?;

    let end = Instant::now();
    println!("Took: {:.2}s", end.duration_since(start).as_secs_f32());

    images[0].save("image.png")?;

    Ok(())
}
