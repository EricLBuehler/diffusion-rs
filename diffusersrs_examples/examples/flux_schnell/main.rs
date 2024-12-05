use diffusersrs_core::{Pipeline, TokenSource};

fn main() -> anyhow::Result<()> {
    let pipeline = Pipeline::load(
        "black-forest-labs/FLUX.1-schnell".to_string(),
        false,
        TokenSource::CacheToken,
        None,
    )?;

    Ok(())
}
