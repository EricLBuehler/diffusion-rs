use crate::core::{Result, Tensor};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct ArgSort {
    asc: bool,
    last_dim: usize,
}

impl ArgSort {
    fn asort<T: crate::core::WithDType>(&self, vs: &[T], layout: &crate::core::Layout) -> Vec<u32> {
        #[allow(clippy::uninit_vec)]
        // Safety: indexes are set later in the parallelized section.
        let mut sort_indexes = unsafe {
            let el_count = layout.shape().elem_count();
            let mut v = Vec::with_capacity(el_count);
            v.set_len(el_count);
            v
        };
        if self.asc {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&i, &j| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        } else {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&j, &i| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        }
        sort_indexes
    }
}

impl crate::core::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "argsort"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::core::CpuStorage,
        layout: &crate::core::Layout,
    ) -> Result<(crate::core::CpuStorage, crate::core::Shape)> {
        let sort_indexes = match storage {
            crate::core::CpuStorage::U8(vs) => self.asort(vs, layout),
            crate::core::CpuStorage::I8(vs) => self.asort(vs, layout),
            crate::core::CpuStorage::U32(vs) => self.asort(vs, layout),
            crate::core::CpuStorage::I16(vs) => self.asort(vs, layout),
            crate::core::CpuStorage::I32(vs) => self.asort(vs, layout),
            crate::core::CpuStorage::I64(vs) => self.asort(vs, layout),
            crate::core::CpuStorage::BF16(vs) => self.asort(vs, layout),
            crate::core::CpuStorage::F16(vs) => self.asort(vs, layout),
            crate::core::CpuStorage::F32(vs) => self.asort(vs, layout),
            crate::core::CpuStorage::F64(vs) => self.asort(vs, layout),
            crate::core::CpuStorage::F8E4M3(vs) => self.asort(vs, layout),
        };
        let sort_indexes = crate::core::CpuStorage::U32(sort_indexes);
        Ok((sort_indexes, layout.shape().into()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &crate::core::CudaStorage,
        layout: &crate::core::Layout,
    ) -> Result<(crate::core::CudaStorage, crate::core::Shape)> {
        use crate::core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits,
        };
        use crate::core::cuda_backend::{
            kernel_name, kernels, CudaStorageSlice as S, Map1Any, WrapErr,
        };
        use crate::core::{CudaDevice, WithDType};

        #[allow(non_local_definitions)]
        impl Map1Any for ArgSort {
            fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(CudaSlice<T>) -> S>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &crate::core::Layout,
                _wrap: W,
            ) -> Result<S> {
                let slice = match layout.contiguous_offsets() {
                    None => crate::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let elem_count = layout.shape().elem_count();
                let dst = unsafe { dev.alloc::<u32>(elem_count) }.w()?;
                let func = if self.asc {
                    dev.get_or_load_func(&kernel_name::<T>("asort_asc"), kernels::SORT)?
                } else {
                    dev.get_or_load_func(&kernel_name::<T>("asort_desc"), kernels::SORT)?
                };
                let ncols = self.last_dim;
                let nrows = elem_count / ncols;
                let ncols_pad = next_power_of_2(ncols);
                let params = (&slice, &dst, ncols as i32, ncols_pad as i32);
                let cfg = LaunchConfig {
                    grid_dim: (1, nrows as u32, 1),
                    block_dim: (ncols_pad as u32, 1, 1),
                    shared_mem_bytes: (ncols_pad * std::mem::size_of::<u32>()) as u32,
                };
                unsafe { func.launch(cfg, params) }.w()?;
                Ok(S::U32(dst))
            }
        }

        use crate::core::backend::BackendStorage;
        let dev = storage.device();
        let slice = self.map(&storage.slice, dev, layout)?;
        let dst = crate::core::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &crate::core::MetalStorage,
        layout: &crate::core::Layout,
    ) -> Result<(crate::core::MetalStorage, crate::core::Shape)> {
        use crate::core::backend::BackendStorage;
        use crate::core::DType;

        let name = {
            if self.asc {
                match storage.dtype() {
                    DType::BF16 => "asort_asc_bf16",
                    DType::F16 => "asort_asc_f16",
                    DType::F32 => "asort_asc_f32",
                    DType::F64 => "asort_asc_f64",
                    DType::I8 => "asort_asc_i8",
                    DType::U8 => "asort_asc_u8",
                    DType::U32 => "asort_asc_u32",
                    DType::I64 => "asort_asc_i64",
                    DType::I32 => "asort_asc_i32",
                    DType::I16 => "asort_asc_i16",
                    DType::F8E4M3 => crate::bail!("Metal device does not yet support F8E4M3."),
                }
            } else {
                match storage.dtype() {
                    DType::BF16 => "asort_desc_bf16",
                    DType::F16 => "asort_desc_f16",
                    DType::F32 => "asort_desc_f32",
                    DType::F64 => "asort_desc_f64",
                    DType::I8 => "asort_desc_i8",
                    DType::U8 => "asort_desc_u8",
                    DType::U32 => "asort_desc_u32",
                    DType::I64 => "asort_desc_i64",
                    DType::I32 => "asort_desc_i32",
                    DType::I16 => "asort_desc_i16",
                    DType::F8E4M3 => crate::bail!("Metal device does not yet support F8E4M3."),
                }
            }
        };
        let device = storage.device();
        let kernels = device.kernels();
        let command_buffer = device.command_buffer()?;
        let el = layout.shape().elem_count();
        let ncols = self.last_dim;
        let nrows = el / ncols;
        let src = crate::core::metal_backend::buffer_o(storage.buffer(), layout, storage.dtype());
        let dst = device.new_buffer(el, DType::U32, "asort")?;
        let mut ncols_pad = 1;
        while ncols_pad < ncols {
            ncols_pad *= 2;
        }
        crate::metal_kernels::call_arg_sort(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            nrows,
            ncols,
            ncols_pad,
            src,
            &dst,
        )
        .map_err(crate::core::Error::wrap)?;
        let dst = crate::core::MetalStorage::new(dst, device.clone(), el, DType::U32);
        Ok((dst, layout.shape().clone()))
    }
}

#[allow(unused)]
fn next_power_of_2(x: usize) -> usize {
    let mut n = 1;
    while n < x {
        n *= 2
    }
    n
}

impl Tensor {
    /// Returns the indices that sort the tensor along the last dimension.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn arg_sort_last_dim(&self, asc: bool) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(crate::core::Error::RequiresContiguous {
                op: "arg_sort_last_dim",
            });
        }
        let last_dim = match self.dims().last() {
            None => crate::bail!("empty last-dim in arg-sort"),
            Some(last_dim) => *last_dim,
        };
        // No need for a backward pass for arg sort.
        self.apply_op1_no_bwd(&ArgSort { asc, last_dim })
    }

    /// Sorts the tensor along the last dimension, returns the sorted tensor together with the
    /// sorted indexes.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn sort_last_dim(&self, asc: bool) -> Result<(Tensor, Tensor)> {
        if !self.is_contiguous() {
            return Err(crate::core::Error::RequiresContiguous {
                op: "sort_last_dim",
            });
        }
        let asort = self.arg_sort_last_dim(asc)?;
        let sorted = self.gather(&asort, crate::core::D::Minus1)?;
        Ok((sorted, asort))
    }
}
