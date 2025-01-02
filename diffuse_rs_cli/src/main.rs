use cliclack::input;
use std::{path::PathBuf, time::Instant};

use clap::{Parser, Subcommand};
use diffuse_rs_core::{DiffusionGenerationParams, ModelSource, Pipeline, TokenSource};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

const GUIDANCE_SCALE_DEFAULT: f64 = 0.0;

#[derive(Debug, Subcommand)]
pub enum SourceCommand {
    /// Load the model from a DDUF file.
    Dduf {
        /// DDUF file path
        #[arg(short, long)]
        file: String,
    },

    /// Load the model from some model ID (local path or Hugging Face model ID)
    ModelId {
        /// Model ID
        #[arg(short, long)]
        model_id: String,
    },
}

#[derive(Parser)]
struct Args {
    #[clap(subcommand)]
    source: SourceCommand,

    /// Hugging Face token. Useful for accessing gated repositories.
    /// By default, the Hugging Face token at ~/.cache/huggingface/token is used.
    #[arg(long)]
    token: Option<String>,

    /// Guidance scale to use. This is model specific. If not specified, defaults to 0.0.
    #[arg(short, long)]
    scale: Option<f64>,

    /// Number of denoising steps. This is model specific. A higher number of steps often means higher quality.
    #[arg(short, long)]
    num_steps: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let source = match args.source {
        SourceCommand::Dduf { file } => ModelSource::dduf(file)?,
        SourceCommand::ModelId { model_id } => ModelSource::from_model_id(model_id),
    };
    let token = args
        .token
        .map(TokenSource::Literal)
        .unwrap_or(TokenSource::CacheToken);

    let pipeline = Pipeline::load(source, false, token, None)?;

    let height: usize = input("Height:")
        .default_input("720")
        .validate(|input: &String| {
            if input.parse::<usize>().map_err(|e| e.to_string())? == 0 {
                Err("Nonzero value is required!".to_string())
            } else {
                Ok(())
            }
        })
        .interact()?;
    let width: usize = input("Width:")
        .default_input("1280")
        .validate(|input: &String| {
            if input.parse::<usize>().map_err(|e| e.to_string())? == 0 {
                Err("Nonzero value is required!".to_string())
            } else {
                Ok(())
            }
        })
        .interact()?;

    loop {
        let prompt: String = input("Prompt:")
            .validate(|input: &String| {
                if input.is_empty() {
                    Err("Prompt is required!")
                } else {
                    Ok(())
                }
            })
            .interact()?;

        let start = Instant::now();

        let images = pipeline.forward(
            vec![prompt],
            DiffusionGenerationParams {
                height,
                width,
                num_steps: args.num_steps,
                guidance_scale: args.scale.unwrap_or(GUIDANCE_SCALE_DEFAULT),
            },
        )?;

        let end = Instant::now();
        println!(
            "Image generation took: {:.2}s",
            end.duration_since(start).as_secs_f32()
        );

        let out_file: String = input("Save image to:")
            .validate(|input: &String| {
                if input.is_empty() {
                    Err("Image path is required!")
                } else {
                    let path = PathBuf::from(input);
                    let ext = path.extension().ok_or("Extension is required!")?;
                    if !["png", "jpg"].contains(&ext.to_str().unwrap()) {
                        Err(".png or .jpg extension is required!")
                    } else {
                        Ok(())
                    }
                }
            })
            .interact()?;

        images[0].save(out_file)?;
    }
}
