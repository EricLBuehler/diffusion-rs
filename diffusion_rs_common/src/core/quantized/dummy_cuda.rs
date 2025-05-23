#![allow(unused)]
use super::GgmlDType;
use crate::core::{CudaDevice, CudaStorage, Error, Result};

pub struct QCudaStorage {
    dtype: GgmlDType,
    device: CudaDevice,
}

impl QCudaStorage {
    pub fn zeros(_: &CudaDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn dequantize_f16(&self, _elem_count: usize) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize(&mut self, _src: &CudaStorage) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &CudaStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::core::CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize_onto(&mut self, _src: &crate::core::CpuStorage) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::core::Shape,
        _storage: &CudaStorage,
        _layout: &crate::core::Layout,
    ) -> Result<(CudaStorage, crate::core::Shape)> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &CudaDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithCudaSupport)
}
