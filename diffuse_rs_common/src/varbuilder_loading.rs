//! Utilities for creating a VarBuilder from a VarMap loaded from tensor storage formats.

use std::{
    collections::HashMap,
    thread::{self, JoinHandle},
};

use crate::{
    core::{safetensors::MmapedSafetensors, DType, Device, Result, Tensor},
    ModelSource,
};
use crate::{
    safetensors::BytesSafetensors,
    varbuilder::{SimpleBackend, VarBuilderArgs},
    FileData, VarBuilder,
};

use super::progress::IterWithProgress;

trait TensorLoaderBackend {
    fn get_names(&self) -> Vec<String>;
    fn load_name(&self, name: &str, device: &Device, dtype: Option<DType>) -> Result<Tensor>;
}

struct SafetensorBackend(MmapedSafetensors);

impl TensorLoaderBackend for SafetensorBackend {
    fn get_names(&self) -> Vec<String> {
        self.0
            .tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>()
    }
    fn load_name(&self, name: &str, device: &Device, dtype: Option<DType>) -> Result<Tensor> {
        let t = self.0.load(name, device)?;
        if let Some(dtype) = dtype {
            t.to_dtype(dtype)
        } else {
            Ok(t)
        }
    }
}

struct BytesSafetensorBackend<'a>(BytesSafetensors<'a>);

impl TensorLoaderBackend for BytesSafetensorBackend<'_> {
    fn get_names(&self) -> Vec<String> {
        self.0
            .tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>()
    }
    fn load_name(&self, name: &str, device: &Device, dtype: Option<DType>) -> Result<Tensor> {
        let t = self.0.load(name, device)?;
        if let Some(dtype) = dtype {
            t.to_dtype(dtype)
        } else {
            Ok(t)
        }
    }
}

/// Load tensors into a VarBuilder backed by a VarMap using MmapedSafetensors.
/// Set `silent` to not show a progress bar.
///
/// # Predicate semantics:
/// - If `regexes` is specified, this will be used in `make_dummy_predicate` based on `.any`
/// - Otherwise, only include keys for which predicate evaluates to true.
pub fn from_mmaped_safetensors<'a>(
    paths: Vec<FileData>,
    dtype: Option<DType>,
    device: &Device,
    silent: bool,
    src: &ModelSource,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>> {
    // #[allow(clippy::type_complexity)]
    // let mut handles: Vec<JoinHandle<Result<HashMap<String, Tensor>>>> = Vec::new();

    // for path in paths {
    //     let device = device.clone();
    //     let loader = Common;
    //     handles.push(thread::spawn(Box::new(move || {
    //         loader.load_tensors_from_path(&path, &device, dtype, silent, src)
    //     })));
    // }

    // let mut ws = HashMap::new();
    // // Wait until all spawned threads have finished loading tensors:
    // while !handles.iter().all(|h| h.is_finished()) {}
    // for h in handles {
    //     ws.extend(h.join().unwrap()?);
    // }

    let mut ws = HashMap::new();
    for path in paths {
        let device = device.clone();
        let loader = Common;
        ws.extend(loader.load_tensors_from_path(&path, &device, dtype, silent, src)?);
    }

    let first_dtype = DType::BF16; //ws.values().next().unwrap().dtype();
    Ok(VarBuilder::from_tensors(
        ws,
        dtype.unwrap_or(first_dtype),
        device,
    ))
}

trait LoadTensors {
    fn load_tensors_from_path(
        &self,
        path: &FileData,
        device: &Device,
        dtype: Option<DType>,
        silent: bool,
        src: &ModelSource,
    ) -> Result<HashMap<String, Tensor>> {
        let tensors: Box<dyn TensorLoaderBackend> = match path
            .extension()
            .expect("Expected extension")
            .to_str()
            .expect("Expected to convert")
        {
            "safetensors" => match path {
                FileData::Dduf { name: _, start, end } => {
                    let ModelSource::Dduf { file, name: _ } = src else {
                        crate::bail!("expected dduf model source!");
                    };
                    Box::new(BytesSafetensorBackend(BytesSafetensors::new(&file.get_ref()[*start..*end])?))
                }
                FileData::DdufOwned { name: _, data } => {
                    Box::new(BytesSafetensorBackend(BytesSafetensors::new(&data)?))
                }
                FileData::Path(path) => {Box::new(SafetensorBackend(unsafe {
                    crate::core::safetensors::MmapedSafetensors::new(path)?
                }))}
            },
            other => crate::bail!("Unexpected extension `{other}`, this should have been handles by `get_model_paths`."),
        };

        let mut loaded_tensors = HashMap::new();
        for name in tensors.get_names().into_iter().with_progress(silent) {
            let tensor = tensors.load_name(&name, device, dtype)?;

            loaded_tensors.insert(name, tensor);
        }

        Ok(loaded_tensors)
    }
}

struct Common;
impl LoadTensors for Common {}
