use diffusers_core::{DiffusionGenerationParams, ModelSource, Pipeline, TokenSource};

use clap::{Parser, ValueEnum};

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
    prompt: String
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let model_id = match args.which {
        Which::Dev => "black-forest-labs/FLUX.1-dev",
        Which::Schnell => "black-forest-labs/FLUX.1-schnell",
    };

    let pipeline = Pipeline::load(
        ModelSource::from_model_id(model_id),
        false,
        TokenSource::CacheToken,
        None,
    )?;

    let images = pipeline.forward(
        vec![args.prompt.to_string()],
        DiffusionGenerationParams::default(),
    )?;
    images[0].save("image.png")?;

    Ok(())
}
