use candle_core::safetensors::Load;
use candle_core::{Device, Error, Result, Tensor};
use safetensors::tensor as st;
use safetensors::tensor::SafeTensors;

pub struct BytesSafetensors<'a> {
    safetensors: SafeTensors<'a>,
}

impl<'a> BytesSafetensors<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<BytesSafetensors<'a>> {
        let st = safetensors::SafeTensors::deserialize(bytes).map_err(|e| Error::from(e))?;
        Ok(Self { safetensors: st })
    }

    pub fn load(&self, name: &str, dev: &Device) -> Result<Tensor> {
        self.get(name)?.load(dev)
    }

    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        self.safetensors.tensors()
    }

    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        Ok(self.safetensors.tensor(name)?)
    }
}
