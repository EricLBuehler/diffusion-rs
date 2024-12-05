use diffusersrs_core::{DiffusionGenerationParams, Pipeline, TokenSource};

fn main() -> anyhow::Result<()> {
    let pipeline = Pipeline::load(
        "black-forest-labs/FLUX.1-schnell".to_string(),
        false,
        TokenSource::CacheToken,
        None,
    )?;

    let images = pipeline.forward(
        vec!["Draw a picture of a computer.".to_string()],
        DiffusionGenerationParams::default(),
    )?;
    images[0].save("image.png")?;

    Ok(())
}
