//! A `VarBuilder` is used to retrieve variables used by a model. These variables can either come
//! from a pre-trained checkpoint, e.g. using `VarBuilder::from_mmaped_safetensors`, or initialized
//! for training, e.g. using `VarBuilder::from_varmap`.
use candle_core::{DType, Device, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// A structure used to retrieve variables, these variables can either come from storage or be
/// generated via some form of initialization.
///
/// The way to retrieve variables is defined in the backend embedded in the `VarBuilder`.
pub struct VarBuilderArgs<'a, B: Backend> {
    data: Arc<TensorData<B>>,
    path: Vec<String>,
    pub dtype: DType,
    _phantom: std::marker::PhantomData<&'a B>,
}

impl<B: Backend> Clone for VarBuilderArgs<'_, B> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            path: self.path.clone(),
            dtype: self.dtype,
            _phantom: self._phantom,
        }
    }
}

/// A simple `VarBuilder`, this is less generic than `VarBuilderArgs` but should cover most common
/// use cases.
pub type VarBuilder<'a> = VarBuilderArgs<'a, Box<dyn SimpleBackend + 'a>>;

struct TensorData<B: Backend> {
    backend: Arc<B>,
    pub dtype: DType,
    pub device: Device,
}

/// A trait that defines how tensor data is retrieved.
///
/// Typically this would use disk storage in some specific format, or random initialization.
/// Note that there is a specialized version of this trait (`SimpleBackend`) that can be used most
/// of the time. The main restriction is that it doesn't allow for specific args (besides
/// initialization hints).
pub trait Backend: Send + Sync {
    type Hints: Default;

    /// Retrieve a tensor with some target shape.
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Self::Hints,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor>;

    /// Retrieve a tensor based on the name.
    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor>;

    fn contains_tensor(&self, name: &str) -> bool;
}

pub trait SimpleBackend: Send + Sync {
    /// Retrieve a tensor based on a target name and shape.
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor>;

    /// Retrieve a tensor based on the name.
    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor>;

    fn contains_tensor(&self, name: &str) -> bool;
}

impl Backend for Box<dyn SimpleBackend + '_> {
    type Hints = candle_nn::Init;
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Self::Hints,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        self.as_ref().get(s, name, h, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        self.as_ref().get_unchecked(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.as_ref().contains_tensor(name)
    }
}

impl<B: Backend> VarBuilderArgs<'_, B> {
    pub fn new_with_args(backend: B, dtype: DType, dev: &Device) -> Self {
        let data = TensorData {
            backend: Arc::new(backend),
            dtype,
            device: dev.clone(),
        };
        Self {
            data: Arc::new(data),
            path: vec![],
            dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the prefix of the `VarBuilder`.
    pub fn prefix(&self) -> String {
        self.path.join(".")
    }

    /// Returns a new `VarBuilder` using the root path.
    pub fn root(&self) -> Self {
        Self {
            data: self.data.clone(),
            path: vec![],
            dtype: self.dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns a new `VarBuilder` with the prefix set to `prefix`.
    pub fn set_prefix(&self, prefix: impl ToString) -> Self {
        Self {
            data: self.data.clone(),
            path: vec![prefix.to_string()],
            dtype: self.dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Return a new `VarBuilder` adding `s` to the current prefix. This can be think of as `cd`
    /// into a directory.
    pub fn push_prefix<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
            dtype: self.dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Short alias for `push_prefix`.
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        self.push_prefix(s)
    }

    /// The device used by default.
    pub fn device(&self) -> &Device {
        &self.data.device
    }

    /// The dtype used by default.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Clone the VarBuilder tweaking its dtype
    pub fn to_dtype(&self, dtype: DType) -> Self {
        Self {
            data: self.data.clone(),
            path: self.path.clone(),
            dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    /// This returns true only if a tensor with the passed in name is available. E.g. when passed
    /// `a`, true is returned if `prefix.a` exists but false is returned if only `prefix.a.b`
    /// exists.
    pub fn contains_tensor(&self, tensor_name: &str) -> bool {
        let path = self.path(tensor_name);
        self.data.backend.contains_tensor(&path)
    }

    /// Retrieve the tensor associated with the given name at the current path.
    pub fn get_with_hints<S: Into<Shape>>(
        &self,
        s: S,
        name: &str,
        hints: B::Hints,
    ) -> Result<Tensor> {
        self.get_with_hints_dtype(s, name, hints, self.data.dtype)
    }

    /// Retrieve the tensor associated with the given name at the current path.
    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Tensor> {
        self.get_with_hints(s, name, Default::default())
    }

    /// Retrieve the tensor associated with the given name at the current path.
    pub fn get_unchecked(&self, name: &str) -> Result<Tensor> {
        self.get_unchecked_dtype(name, self.data.dtype)
    }

    /// Retrieve the tensor associated with the given name & dtype at the current path.
    pub fn get_unchecked_dtype(&self, name: &str, dtype: DType) -> Result<Tensor> {
        let name = self.path(name);
        self.data
            .backend
            .get_unchecked(&name, dtype, &self.data.device)
    }

    /// Retrieve the tensor associated with the given name & dtype at the current path.
    pub fn get_with_hints_dtype<S: Into<Shape>>(
        &self,
        s: S,
        name: &str,
        hints: B::Hints,
        dtype: DType,
    ) -> Result<Tensor> {
        let path = self.path(name);
        self.data
            .backend
            .get(s.into(), &path, hints, dtype, &self.data.device)
    }

    /// Set the device of the VarBuilder.
    pub fn set_device(self, device: Device) -> Self {
        Self {
            data: Arc::new(TensorData {
                backend: self.data.backend.clone(),
                dtype: self.data.dtype,
                device,
            }),
            ..self
        }
    }

    /// Set the dtype of the VarBuilder.
    pub fn set_dtype(self, dtype: DType) -> Self {
        Self {
            data: Arc::new(TensorData {
                backend: self.data.backend.clone(),
                dtype,
                device: self.data.device.clone(),
            }),
            dtype,
            ..self
        }
    }
}

struct Zeros;

impl SimpleBackend for Zeros {
    fn get(
        &self,
        s: Shape,
        _: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        Tensor::zeros(s, dtype, dev)
    }

    fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
        candle_core::bail!(
            "`Zeros` requires a shape for tensor retrieval, use `get` instead of `get_unchecked`"
        )
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

impl SimpleBackend for HashMap<String, Tensor> {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, dev)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        let tensor = self
            .get(name)
            .ok_or_else(|| {
                Error::CannotFindTensor {
                    path: name.to_string(),
                }
                .bt()
            })?
            .clone();
        tensor.to_device(dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.contains_key(name)
    }
}

impl SimpleBackend for candle_core::safetensors::MmapedSafetensors {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, dev)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        self.load(name, dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.get(name).is_ok()
    }
}

impl SimpleBackend for candle_core::safetensors::BufferedSafetensors {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, dev)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        self.load(name, dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.get(name).is_ok()
    }
}

impl SimpleBackend for candle_core::safetensors::SliceSafetensors<'_> {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, dev)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        self.load(name, dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.get(name).is_ok()
    }
}

impl<'a> VarBuilder<'a> {
    /// Initializes a `VarBuilder` using a custom backend.
    ///
    /// It is preferred to use one of the more specific constructors. This
    /// constructor is provided to allow downstream users to define their own
    /// backends.
    pub fn from_backend(
        backend: Box<dyn SimpleBackend + 'a>,
        dtype: DType,
        device: Device,
    ) -> Self {
        let data = TensorData {
            backend: Arc::new(backend),
            dtype,
            device,
        };
        Self {
            data: Arc::new(data),
            path: vec![],
            dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Initializes a `VarBuilder` that uses zeros for any tensor.
    pub fn zeros(dtype: DType, dev: &Device) -> Self {
        Self::from_backend(Box::new(Zeros), dtype, dev.clone())
    }

    /// Initializes a `VarBuilder` that retrieves tensors stored in a hashtable. An error is
    /// returned if no tensor is available under the requested path or on shape mismatches.
    pub fn from_tensors(ts: HashMap<String, Tensor>, dtype: DType, dev: &Device) -> Self {
        Self::from_backend(Box::new(ts), dtype, dev.clone())
    }

    /// Initializes a `VarBuilder` that retrieves tensors stored in a collection of safetensors
    /// files.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn from_mmaped_safetensors<P: AsRef<std::path::Path>>(
        paths: &[P],
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let tensors = candle_core::safetensors::MmapedSafetensors::multi(paths)?;
        Ok(Self::from_backend(Box::new(tensors), dtype, dev.clone()))
    }

    /// Initializes a `VarBuilder` from a binary buffer in the safetensor format.
    pub fn from_buffered_safetensors(data: Vec<u8>, dtype: DType, dev: &Device) -> Result<Self> {
        let tensors = candle_core::safetensors::BufferedSafetensors::new(data)?;
        Ok(Self::from_backend(Box::new(tensors), dtype, dev.clone()))
    }

    /// Initializes a `VarBuilder` from a binary slice in the safetensor format.
    pub fn from_slice_safetensors(data: &'a [u8], dtype: DType, dev: &Device) -> Result<Self> {
        let tensors = candle_core::safetensors::SliceSafetensors::new(data)?;
        Ok(Self::from_backend(Box::new(tensors), dtype, dev.clone()))
    }
}
