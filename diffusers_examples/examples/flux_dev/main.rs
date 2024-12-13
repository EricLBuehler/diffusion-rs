use diffusers_core::{DiffusionGenerationParams, ModelSource, Pipeline, TokenSource};

fn main() -> anyhow::Result<()> {
    let pipeline = Pipeline::load(
        ModelSource::from_model_id("black-forest-labs/FLUX.1-dev"),
        false,
        TokenSource::CacheToken,
        None,
    )?;

    let images = pipeline.forward(
        vec!["Draw a picture of a beautiful sunset in the winter in the mountains, 4k, high quality.".to_string()],
        DiffusionGenerationParams::default(),
    )?;
    images[0].save("image_dev.png")?;

    Ok(())
}
