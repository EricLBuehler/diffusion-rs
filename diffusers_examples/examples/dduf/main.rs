use clap::Parser;
use diffusers_core::{DiffusionGenerationParams, ModelSource, Pipeline, TokenSource};

#[derive(Parser)]
struct Args {
    /// DDUF file
    #[arg(long, short)]
    file: String,

    /// Prompt to use
    #[arg(short, long)]
    prompt: String
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let pipeline = Pipeline::load(
        ModelSource::dduf(args.file)?,
        false,
        TokenSource::CacheToken,
        None,
    )?;

    let images = pipeline.forward(
        vec![args.prompt],
        DiffusionGenerationParams::default(),
    )?;
    images[0].save("image.png")?;

    Ok(())
}
