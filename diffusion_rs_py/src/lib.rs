use std::io::Cursor;

use pyo3::{
    pyclass, pymethods, pymodule,
    types::{PyBytes, PyModule, PyModuleMethods},
    Bound, Py, PyResult, Python,
};

fn wrap_anyhow_error(e: anyhow::Error) -> pyo3::PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Offloading {
    Full,
}

#[pyclass]
#[derive(Clone, Debug)]
pub enum ModelSource {
    ModelId { model_id: String },
    DdufFile { file: String },
}

#[pyclass]
#[pyo3(get_all)]
#[derive(Clone, Debug)]
pub struct DiffusionGenerationParams {
    pub height: usize,
    pub width: usize,
    pub num_steps: usize,
    pub guidance_scale: f64,
}

#[pyclass(eq, eq_int)]
#[pyo3(get_all)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelDType {
    Auto,
    BF16,
    F16,
    F32,
}

#[pymethods]
impl DiffusionGenerationParams {
    #[new]
    #[pyo3(signature = (
        height,
        width,
        num_steps,
        guidance_scale,
    ))]
    pub fn new(
        height: usize,
        width: usize,
        num_steps: usize,
        guidance_scale: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            height,
            width,
            num_steps,
            guidance_scale,
        })
    }

    pub fn __repr__(&self) -> String {
        format!("DiffusionGenerationParams(height = {}, width = {}, num_steps = {}, guidance_scale = {})", self.height,self.width,self.num_steps,self.guidance_scale)
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass]
pub struct Pipeline(diffusion_rs_core::Pipeline);

#[pymethods]
impl Pipeline {
    #[new]
    #[pyo3(signature = (
        source,
        silent = false,
        token = None,
        revision = None,
        offloading = None,
        dtype = ModelDType::Auto,
    ))]
    pub fn new(
        source: ModelSource,
        silent: bool,
        token: Option<String>,
        revision: Option<String>,
        offloading: Option<Offloading>,
        dtype: ModelDType,
    ) -> PyResult<Self> {
        let token = token
            .map(diffusion_rs_core::TokenSource::Literal)
            .unwrap_or(diffusion_rs_core::TokenSource::CacheToken);
        let source = match source {
            ModelSource::DdufFile { file } => {
                diffusion_rs_core::ModelSource::dduf(file).map_err(wrap_anyhow_error)?
            }
            ModelSource::ModelId { model_id } => {
                diffusion_rs_core::ModelSource::from_model_id(model_id)
            }
        };
        let offloading = offloading.map(|offloading| match offloading {
            Offloading::Full => diffusion_rs_core::Offloading::Full,
        });
        let dtype = match dtype {
            ModelDType::Auto => diffusion_rs_core::ModelDType::Auto,
            ModelDType::F16 => diffusion_rs_core::ModelDType::F16,
            ModelDType::BF16 => diffusion_rs_core::ModelDType::BF16,
            ModelDType::F32 => diffusion_rs_core::ModelDType::F32,
        };
        Ok(Self(
            diffusion_rs_core::Pipeline::load(source, silent, token, revision, offloading, &dtype)
                .map_err(wrap_anyhow_error)?,
        ))
    }

    fn forward(
        &self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        let images = self
            .0
            .forward(
                prompts,
                diffusion_rs_core::DiffusionGenerationParams {
                    height: params.height,
                    width: params.width,
                    num_steps: params.num_steps,
                    guidance_scale: params.guidance_scale,
                },
            )
            .map_err(wrap_anyhow_error)?;

        let mut images_bytes = Vec::new();
        for image in images {
            let mut buf = Vec::new();
            image
                .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
                .unwrap();
            let bytes: Py<PyBytes> = Python::with_gil(move |py| PyBytes::new(py, &buf).into());
            images_bytes.push(bytes);
        }

        Ok(images_bytes)
    }
}

#[pymodule]
fn diffusion_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ModelSource>()?;
    m.add_class::<DiffusionGenerationParams>()?;
    m.add_class::<Pipeline>()?;
    Ok(())
}
