mod flux;

use std::{collections::HashMap, fmt::Display, sync::Arc};

use anyhow::Result;
use candle_core::{Device, Tensor};
use flux::FluxLoader;
use image::{DynamicImage, RgbImage};
use serde::Deserialize;

use diffusers_common::{FileData, FileLoader, ModelSource, NiceProgressBar, TokenSource};
use tracing::info;

#[derive(Debug, Clone)]
pub struct DiffusionGenerationParams {
    pub height: usize,
    pub width: usize,
    /// The number of denoising steps. More denoising steps usually lead to a higher quality image at the
    /// expense of slower inference but depends on the model being used.
    pub num_steps: usize,
    /// Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
    /// usually at the expense of lower image quality. Defaults to 3.5.
    pub guidance_scale: f64,
}

#[derive(Debug)]
pub(crate) enum ComponentElem {
    Model {
        safetensors: HashMap<String, FileData>,
        config: FileData,
    },
    Config {
        files: HashMap<String, FileData>,
    },
    Other {
        files: HashMap<String, FileData>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComponentName {
    Scheduler,
    TextEncoder(usize),
    Tokenizer(usize),
    Transformer,
    Vae,
}

impl Display for ComponentName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scheduler => write!(f, "scheduler"),
            Self::Transformer => write!(f, "transformer"),
            Self::Vae => write!(f, "vae"),
            Self::TextEncoder(1) => write!(f, "text_encoder"),
            Self::TextEncoder(x) => write!(f, "text_encoder_{x}"),
            Self::Tokenizer(1) => write!(f, "tokenizer"),
            Self::Tokenizer(x) => write!(f, "tokenizer_{x}"),
        }
    }
}

pub(crate) trait Loader {
    fn name(&self) -> &'static str;
    fn required_component_names(&self) -> Vec<ComponentName>;
    fn load_from_components(
        &self,
        components: HashMap<ComponentName, ComponentElem>,
        device: &Device,
    ) -> Result<Arc<dyn ModelPipeline>>;
}

pub trait ModelPipeline: Send + Sync {
    fn forward(
        &self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
    ) -> candle_core::Result<Tensor>;
}

#[derive(Clone, Debug, Deserialize)]
struct ModelIndex {
    #[serde(rename = "_class_name")]
    name: String,
}

pub struct Pipeline(Arc<dyn ModelPipeline>);

impl Pipeline {
    pub fn load(
        source: ModelSource,
        silent: bool,
        token: TokenSource,
        revision: Option<String>,
    ) -> Result<Self> {
        info!("loading from source: {source}.");

        let mut loader = FileLoader::from_model_source(source, silent, token, revision)?;
        let files = loader.list_files()?;
        let transformer_files = loader.list_transformer_files()?;

        if !files.contains(&"model_index.json".to_string()) {
            anyhow::bail!("Expected `model_index.json` file present.");
        }

        let ModelIndex { name } = serde_json::from_str(
            &loader
                .read_file("model_index.json", false)?
                .read_to_string()?,
        )?;

        let model_loader: Box<dyn Loader> = match name.as_str() {
            "FluxPipeline" => Box::new(FluxLoader),
            other => anyhow::bail!("Unexpected loader type `{other:?}`."),
        };

        info!("model architecture is: {}", model_loader.name());

        let mut components = HashMap::new();
        for component in NiceProgressBar::<_, 'g'>(
            model_loader.required_component_names().into_iter(),
            "Loading components",
        ) {
            let (files, from_transformer, dir) =
                if component == ComponentName::Transformer && transformer_files.is_some() {
                    (transformer_files.clone().unwrap(), true, "".to_string())
                } else {
                    (files.clone(), false, format!("{component}/"))
                };
            let files_for_component = files
                .iter()
                .filter(|file| file.starts_with(&dir))
                .filter(|file| !file.ends_with('/'))
                .cloned()
                .collect::<Vec<_>>();

            // Try to determine the component's type.
            // 1) Model: models contain .safetensors and potentially a config.json
            // 2) Config: general config, a file ends with .json
            // 3) Other: doesn't have safetensors and is not all json
            let component_elem = if files_for_component
                .iter()
                .any(|file| file.ends_with(".safetensors"))
            {
                let mut safetensors = HashMap::new();
                for file in files_for_component
                    .iter()
                    .filter(|file| file.ends_with(".safetensors"))
                {
                    safetensors.insert(file.clone(), loader.read_file(file, from_transformer)?);
                }
                ComponentElem::Model {
                    safetensors,
                    config: loader.read_file(&format!("{dir}config.json"), from_transformer)?,
                }
            } else if files_for_component
                .iter()
                .all(|file| file.ends_with(".json"))
            {
                let mut files = HashMap::new();
                for file in files_for_component
                    .iter()
                    .filter(|file| file.ends_with(".json"))
                {
                    files.insert(file.clone(), loader.read_file(file, from_transformer)?);
                }
                ComponentElem::Config { files }
            } else {
                let mut files = HashMap::new();
                for file in files_for_component {
                    files.insert(file.clone(), loader.read_file(&file, from_transformer)?);
                }
                ComponentElem::Other { files }
            };
            components.insert(component, component_elem);
        }

        #[cfg(not(feature = "metal"))]
        let device = Device::cuda_if_available(0)?;
        #[cfg(feature = "metal")]
        let device = Device::new_metal(0)?;

        let model = model_loader.load_from_components(components, &device)?;

        Ok(Self(model))
    }

    pub fn forward(
        &self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
    ) -> anyhow::Result<Vec<DynamicImage>> {
        let img = self.0.forward(prompts, params)?;

        let (_b, c, h, w) = img.dims4()?;
        let mut images = Vec::new();
        for b_img in img.chunk(img.dim(0)?, 0)? {
            let flattened = b_img.squeeze(0)?.permute((1, 2, 0))?.flatten_all()?;
            if c != 3 {
                anyhow::bail!("Expected 3 channels in image output");
            }
            #[allow(clippy::cast_possible_truncation)]
            images.push(DynamicImage::ImageRgb8(
                RgbImage::from_raw(w as u32, h as u32, flattened.to_vec1::<u8>()?).ok_or(
                    candle_core::Error::Msg("RgbImage has invalid capacity.".to_string()),
                )?,
            ));
        }
        Ok(images)
    }
}
