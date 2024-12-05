//! Utilities for creating a VarBuilder from a VarMap loaded from tensor storage formats.

use std::{
    collections::HashMap,
    path::PathBuf,
    thread::{self, JoinHandle},
};

use candle_core::{
    pickle::PthTensors, safetensors::MmapedSafetensors, DType, Device, Result, Tensor,
};
use candle_nn::{
    var_builder::{SimpleBackend, VarBuilderArgs},
    VarBuilder,
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

struct PickleBackend(PthTensors);

impl TensorLoaderBackend for PickleBackend {
    fn get_names(&self) -> Vec<String> {
        self.0.tensor_infos().keys().cloned().collect::<Vec<_>>()
    }
    fn load_name(&self, name: &str, device: &Device, dtype: Option<DType>) -> Result<Tensor> {
        let t = self
            .0
            .get(name)?
            .ok_or(candle_core::Error::Msg(format!(
                "Could not load tensor {name}"
            )))?
            .to_device(device)?;
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
pub(crate) fn from_mmaped_safetensors<'a>(
    paths: Vec<PathBuf>,
    dtype: Option<DType>,
    device: &Device,
    silent: bool,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>> {
    #[allow(clippy::type_complexity)]
    let mut handles: Vec<JoinHandle<Result<HashMap<String, Tensor>>>> = Vec::new();

    for path in paths {
        let device = device.clone();
        let loader = Common;
        handles.push(thread::spawn(Box::new(move || {
            loader.load_tensors_from_path(&path, &device, dtype, silent)
        })));
    }

    let mut ws = HashMap::new();
    // Wait until all spawned threads have finished loading tensors:
    while !handles.iter().all(|h| h.is_finished()) {}
    for h in handles {
        ws.extend(h.join().unwrap()?);
    }

    let first_dtype = ws.values().next().unwrap().dtype();
    Ok(VarBuilder::from_tensors(
        ws,
        dtype.unwrap_or(first_dtype),
        device,
    ))
}

trait LoadTensors {
    fn load_tensors_from_path(
        &self,
        path: &PathBuf,
        device: &Device,
        dtype: Option<DType>,
        silent: bool,
    ) -> Result<HashMap<String, Tensor>> {
        let tensors: Box<dyn TensorLoaderBackend> = match path
            .extension()
            .expect("Expected extension")
            .to_str()
            .expect("Expected to convert")
        {
            "safetensors" => Box::new(SafetensorBackend(unsafe {
                candle_core::safetensors::MmapedSafetensors::new(path)?
            })),
            "pth" | "pt" | "bin" => Box::new(PickleBackend(
                candle_core::pickle::PthTensors::new(path, None)?
            )),
            other => candle_core::bail!("Unexpected extension `{other}`, this should have been handles by `get_model_paths`."),
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
