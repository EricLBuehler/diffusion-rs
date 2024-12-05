mod flux;

use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};

use anyhow::Result;
use candle_core::{Device, Tensor};
use flux::FluxLoader;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use image::{DynamicImage, RgbImage};
use serde::Deserialize;

use crate::util::{get_token, TokenSource};

#[derive(Debug, Clone)]
pub struct DiffusionGenerationParams {
    pub height: usize,
    pub width: usize,
    /// The number of denoising steps. More denoising steps usually lead to a higher quality image at the
    /// expense of slower inference. Defaults to 50.
    pub num_steps: Option<usize>,
    /// Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
    /// usually at the expense of lower image quality. Defaults to 3.5.
    pub guidance_scale: f64,
}

impl Default for DiffusionGenerationParams {
    /// Image dimensions will be 720x1280. Default steps for the model will be used.
    fn default() -> Self {
        Self {
            height: 720,
            width: 1280,
            num_steps: None,
            guidance_scale: 3.5,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum ComponentElem {
    Model {
        safetensors: HashMap<String, PathBuf>,
        config: PathBuf,
    },
    Config {
        files: HashMap<String, PathBuf>,
    },
    Other {
        files: HashMap<String, PathBuf>,
    },
}

pub(crate) trait Loader {
    fn required_component_names(&self) -> Vec<&'static str>;
    fn load_from_components(
        &self,
        components: HashMap<String, ComponentElem>,
        device: &Device,
    ) -> Result<Arc<dyn ModelPipeline>>;
}

pub trait ModelPipeline {
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
        model_id: String,
        silent: bool,
        token: TokenSource,
        revision: Option<String>,
    ) -> Result<Self> {
        let api = ApiBuilder::new()
            .with_progress(!silent)
            .with_token(get_token(&token)?)
            .build()?;
        let revision = revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(
            model_id,
            RepoType::Model,
            revision.clone(),
        ));

        let files = api
            .info()
            .map(|repo| {
                repo.siblings
                    .iter()
                    .map(|x| x.rfilename.clone())
                    .collect::<Vec<String>>()
            })
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        dbg!(&files);

        if !files.contains(&"model_index.json".to_string()) {
            anyhow::bail!("Expected `model_index.json` file present.");
        }

        let ModelIndex { name } =
            serde_json::from_str(&fs::read_to_string(api.get("model_index.json")?)?)?;

        let loader = match name.as_str() {
            "FluxPipeline" => Box::new(FluxLoader),
            other => anyhow::bail!("Unexpected loader type `{other:?}`."),
        };
        let mut components = HashMap::new();
        for component in loader.required_component_names() {
            let dir = format!("{component}/");
            let files_for_component = files
                .iter()
                .filter(|file| file.starts_with(&dir))
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
                    safetensors.insert(file.clone(), api.get(file)?);
                }
                ComponentElem::Model {
                    safetensors,
                    config: api.get(&format!("{component}/config.json"))?,
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
                    files.insert(file.clone(), api.get(file)?);
                }
                ComponentElem::Config { files }
            } else {
                let mut files = HashMap::new();
                for file in files_for_component {
                    files.insert(file.clone(), api.get(&file)?);
                }
                ComponentElem::Other { files }
            };
            components.insert(component.to_string(), component_elem);
        }

        let device = Device::new_metal(0)?;

        let model = loader.load_from_components(components, &device)?;

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
