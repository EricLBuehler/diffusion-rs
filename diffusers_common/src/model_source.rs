use std::{
    ffi::OsStr,
    fmt::Debug,
    fs::{self, File},
    io::Cursor,
    path::PathBuf,
};

use crate::{get_token, TokenSource};
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Repo, RepoType,
};
use memmap2::Mmap;
use zip::ZipArchive;

pub enum ModelSource {
    ModelId(String),
    ModelIdWithTransformer {
        model_id: String,
        transformer_model_id: String,
    },
    Dduf {
        file: File,
    },
}

impl ModelSource {
    pub fn from_model_id<S: ToString>(model_id: S) -> Self {
        Self::ModelId(model_id.to_string())
    }

    pub fn override_transformer_model_id<S: ToString>(self, model_id: S) -> anyhow::Result<Self> {
        let Self::ModelId(base_id) = self else {
            anyhow::bail!("Expected model ID for the model source")
        };
        Ok(Self::ModelIdWithTransformer {
            model_id: base_id,
            transformer_model_id: model_id.to_string(),
        })
    }

    pub fn dduf<S: ToString>(filename: S) -> anyhow::Result<Self> {
        Ok(Self::Dduf {
            file: File::open(filename.to_string())?,
        })
    }
}

pub enum FileLoader {
    Api(ApiRepo),
    ApiWithTransformer { base: ApiRepo, transformer: ApiRepo },
    Dduf(ZipArchive<Cursor<Mmap>>),
}

impl FileLoader {
    pub fn from_model_source(
        source: ModelSource,
        silent: bool,
        token: TokenSource,
        revision: Option<String>,
    ) -> anyhow::Result<Self> {
        match source {
            ModelSource::ModelId(model_id) => {
                let api_builder = ApiBuilder::new()
                    .with_progress(!silent)
                    .with_token(get_token(&token)?)
                    .build()?;
                let revision = revision.unwrap_or("main".to_string());
                let api = api_builder.repo(Repo::with_revision(
                    model_id,
                    RepoType::Model,
                    revision.clone(),
                ));

                Ok(Self::Api(api))
            }
            ModelSource::Dduf { file } => {
                let mmap = unsafe { Mmap::map(&file)? };
                let cursor = Cursor::new(mmap);
                Ok(Self::Dduf(ZipArchive::new(cursor)?))
            }
            ModelSource::ModelIdWithTransformer {
                model_id,
                transformer_model_id,
            } => {
                let api_builder = ApiBuilder::new()
                    .with_progress(!silent)
                    .with_token(get_token(&token)?)
                    .build()?;
                let revision = revision.unwrap_or("main".to_string());
                let api = api_builder.repo(Repo::with_revision(
                    model_id,
                    RepoType::Model,
                    revision.clone(),
                ));
                let transformer_api = api_builder.repo(Repo::with_revision(
                    transformer_model_id,
                    RepoType::Model,
                    revision.clone(),
                ));

                Ok(Self::ApiWithTransformer {
                    base: api,
                    transformer: transformer_api,
                })
            }
        }
    }

    pub fn list_files(&mut self) -> anyhow::Result<Vec<String>> {
        match self {
            Self::Api(api)
            | Self::ApiWithTransformer {
                base: api,
                transformer: _,
            } => api
                .info()
                .map(|repo| {
                    repo.siblings
                        .iter()
                        .map(|x| x.rfilename.clone())
                        .collect::<Vec<String>>()
                })
                .map_err(|e| anyhow::Error::msg(e.to_string())),
            Self::Dduf(dduf) => (0..dduf.len())
                .map(|i| {
                    dduf.by_index(i)
                        .map(|x| x.name().to_string())
                        .map_err(|e| anyhow::Error::msg(e.to_string()))
                })
                .collect::<anyhow::Result<Vec<_>>>(),
        }
    }

    pub fn list_transformer_files(&self) -> anyhow::Result<Option<Vec<String>>> {
        match self {
            Self::Api(_) | Self::Dduf(_) => Ok(None),

            Self::ApiWithTransformer {
                base: _,
                transformer: api,
            } => api
                .info()
                .map(|repo| {
                    repo.siblings
                        .iter()
                        .map(|x| x.rfilename.clone())
                        .collect::<Vec<String>>()
                })
                .map(Some)
                .map_err(|e| anyhow::Error::msg(e.to_string())),
        }
    }

    pub fn read_file(&mut self, name: &str, from_transformer: bool) -> anyhow::Result<FileData> {
        if from_transformer && !matches!(self, Self::ApiWithTransformer { .. }) {
            anyhow::bail!("This model source has no transformer files.")
        }

        match (self, from_transformer) {
            (Self::Api(api), false)
            | (
                Self::ApiWithTransformer {
                    base: api,
                    transformer: _,
                },
                false,
            ) => Ok(FileData::Path(
                api.get(name)
                    .map_err(|e| anyhow::Error::msg(e.to_string()))?,
            )),
            (
                Self::ApiWithTransformer {
                    base: api,
                    transformer: _,
                },
                true,
            ) => Ok(FileData::Path(
                api.get(name)
                    .map_err(|e| anyhow::Error::msg(e.to_string()))?,
            )),
            (Self::Api(_), true) => anyhow::bail!("This model source has no transformer files."),
            (Self::Dduf(dduf), _) => {
                let mut file = dduf.by_name(name)?;
                let mut data = Vec::new();
                std::io::copy(&mut file, &mut data)?;
                let name = PathBuf::from(file.name().to_string());
                Ok(FileData::Dduf { name, data })
            }
        }
    }
}

pub enum FileData {
    Path(PathBuf),
    Dduf { name: PathBuf, data: Vec<u8> },
}

impl Debug for FileData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Path(p) => write!(f, "path: {}", p.display()),
            Self::Dduf { name, data: _ } => write!(f, "dduf: {}", name.display()),
        }
    }
}

impl FileData {
    pub fn read_to_string(&self) -> anyhow::Result<String> {
        match self {
            Self::Path(p) => Ok(fs::read_to_string(p)?),
            Self::Dduf { name: _, data } => Ok(String::from_utf8(data.to_vec())?),
        }
    }

    pub fn extension(&self) -> Option<&OsStr> {
        match self {
            Self::Path(p) => p.extension(),
            Self::Dduf { name, data: _ } => name.extension(),
        }
    }
}
