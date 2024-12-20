use anyhow::Context;
use candle_core::{
    backend::BackendStorage,
    cuda::{
        cudarc::driver::{
            self,
            sys::{
                CUgraph, CUgraphDebugDot_flags, CUgraphExec, CUgraphInstantiate_flags, CUgraphNode,
                CUstream, CUstreamCaptureMode,
            },
            DevicePtr, DeviceSlice,
        },
        WrapErr,
    },
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Storage, Tensor,
};
use candle_nn::Module;
use half::{bf16, f16};
use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    ffi::CString,
    marker::PhantomData,
    mem::MaybeUninit,
    path::Path,
    process::Command,
    ptr,
};

use crate::{KernelLaunchParams, Node, NodeData};

/// Copy a tensor inplace from src to dst. This can be used to implement `GraphInput`.
///
/// # Safety
/// It must be ensured that the storage of src can be cast to &mut. So no aliasing across threads.
pub unsafe fn copy_inplace(src: &Tensor, dst: &Tensor, device: &Device) -> candle_core::Result<()> {
    match (&*src.storage_and_layout().0, &*dst.storage_and_layout().0) {
        (Storage::Cuda(src), Storage::Cuda(tgt)) => {
            // What we are really doing:

            // unsafe fn cast_to_mut<T>(r: &T) -> &mut T {
            //     // Cast immutable reference to mutable reference
            //     #[allow(invalid_reference_casting)]
            //     &mut *(r as *const T as *mut T)
            // }
            // let dst = unsafe { cast_to_mut(tgt.as_cuda_slice::<bf16>()?) };
            // cu_device.dtod_copy(src, dst)?;

            if src.dtype() != dst.dtype() {
                candle_core::bail!("DTypes must match!");
            }

            match src.dtype() {
                DType::BF16 => {
                    let tgt = tgt.as_cuda_slice::<bf16>()?;
                    let src = src.as_cuda_slice::<bf16>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<bf16>(),
                    )
                    .w()?;
                }
                DType::F16 => {
                    let tgt = tgt.as_cuda_slice::<f16>()?;
                    let src = src.as_cuda_slice::<f16>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<f16>(),
                    )
                    .w()?;
                }
                DType::F32 => {
                    let tgt = tgt.as_cuda_slice::<f32>()?;
                    let src = src.as_cuda_slice::<f32>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<f32>(),
                    )
                    .w()?;
                }
                DType::F64 => {
                    let tgt = tgt.as_cuda_slice::<f64>()?;
                    let src = src.as_cuda_slice::<f64>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<f64>(),
                    )
                    .w()?;
                }
                DType::I64 => {
                    let tgt = tgt.as_cuda_slice::<i64>()?;
                    let src = src.as_cuda_slice::<i64>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<i64>(),
                    )
                    .w()?;
                }
                DType::U32 => {
                    let tgt = tgt.as_cuda_slice::<u32>()?;
                    let src = src.as_cuda_slice::<u32>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<u32>(),
                    )
                    .w()?;
                }
                DType::U8 => {
                    let tgt = tgt.as_cuda_slice::<u8>()?;
                    let src = src.as_cuda_slice::<u8>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<u8>(),
                    )
                    .w()?;
                }
            }
            device.synchronize()?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

pub enum GraphDumpFormat {
    Svg,
    Png,
    Dot,
}

pub enum GraphDumpVerbosity {
    Clean,
    Verbose,
}

pub trait GraphInput {
    fn load_inputs_inplace(&self, input: Self, device: &Device) -> candle_core::Result<()>;
}

impl GraphInput for HashMap<&'static str, Tensor> {
    fn load_inputs_inplace(&self, input: Self, device: &Device) -> candle_core::Result<()> {
        let mut added = HashSet::new();
        for (name, input) in &input {
            if !added.insert(name) {
                panic!("Got duplicate inputs {name}");
            }
            if !input.is_contiguous() {
                panic!("Input {name} is not contiguous");
            }
            if let Some(inp_ref) = self.get(name) {
                unsafe { copy_inplace(input, inp_ref, device)? };
            } else {
                panic!("Graph has no input {name}");
            }
        }
        if added.len() != input.len() {
            panic!(
                "Some inputs were not provided: expected {:?}, got {added:?}",
                input.keys().collect::<Vec<_>>()
            );
        }
        Ok(())
    }
}

pub struct Graph<T: GraphInput> {
    graph: CUgraph,
    exec: CUgraphExec,
    stream: CUstream,
    device: Device,
    input: T,
    ran_graph: Cell<bool>,
    // CUgraph is not thread safe!
    _marker: PhantomData<*const ()>,
}

impl<T: GraphInput> Graph<T> {
    /// Initialize a CUDA graph, executing the closure to capture a graph.
    ///
    /// The input tensors provided must all be contiguous.
    pub fn new(
        from_code: impl FnOnce(&T) -> candle_core::Result<()>,
        device: &Device,
        input: T,
    ) -> candle_core::Result<Self> {
        let cu_device = match &device {
            Device::Cuda(dev) => dev,
            _ => candle_core::bail!("Must have CUDA device."),
        };

        let cu_stream = cu_device.cu_stream();

        // Initialize all ptx files
        // `load_ptx` cannot be called while capturing the stream so we need this to happen
        // beforehand.
        {
            // Fill
            let x = Tensor::zeros((128, 128), DType::F32, device)?;

            // Affine
            let _ = x.affine(1., 0.5)?;

            // Binary
            let _ = x.mul(&x)?;

            // Cast
            let _ = x.to_dtype(DType::BF16)?;

            // Conv2d
            {
                let ws = Tensor::zeros((3, 3, 4, 4), DType::F32, device)?;
                let conv_xs = Tensor::zeros((1, 3, 48, 48), DType::F32, device)?;
                let _ = conv_xs.conv2d(&ws, 0, 1, 1, 1)?;
            }

            // Indexing
            {
                let indices = Tensor::new(vec![0u32, 2, 4], device)?;
                let _ = x.index_select(&indices, 0)?;
            }

            // FUSED_RMS_NORM
            // TODO

            // FUSED_ROPE
            // TODO

            // Quantized
            {
                let q = QMatMul::from_qtensor(QTensor::quantize(&x, GgmlDType::Q8_0)?)?;
                let _ = q.forward(&x)?;
            }

            // Reduce
            let _ = candle_nn::ops::softmax_last_dim(&x)?;

            // Sort
            let _ = x.sort_last_dim(true)?;

            // Ternary
            let _ = x.to_dtype(DType::U32)?.where_cond(
                &Tensor::new(0f32, device)?.broadcast_as(x.shape())?,
                &Tensor::new(1f32, device)?.broadcast_as(x.shape())?,
            )?;

            // Unary
            let _ = x.neg()?;

            device.synchronize()?;
        }

        let mut cu_graph: CUgraph = unsafe {
            let mut cu_graph = MaybeUninit::uninit();
            driver::sys::lib()
                .cuGraphCreate(cu_graph.as_mut_ptr(), 0)
                .result()
                .w()?;
            cu_graph.assume_init()
        };

        unsafe {
            driver::sys::lib()
                .cuStreamBeginCapture_v2(
                    *cu_stream,
                    CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
                .result()
                .w()?
        }

        from_code(&input)?;

        /////  END CAPTURE AND WRITE TO THE GRAPH
        unsafe {
            driver::sys::lib()
                .cuStreamEndCapture(*cu_stream, &mut cu_graph as *mut _)
                .result()
                .w()?;
        }

        /////  CREATING THE GRAPH EXECUTOR
        let cu_graph_e: CUgraphExec = unsafe {
            let mut cu_graph_e = MaybeUninit::uninit();
            // https://github.com/pytorch/pytorch/blob/c7b0d4b148cf2e4e68f14193549945e1639bff40/aten/src/ATen/cuda/CUDAGraph.cpp#L166-L176
            driver::sys::lib()
                .cuGraphInstantiateWithFlags(
                    cu_graph_e.as_mut_ptr(),
                    cu_graph,
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
                        as u64,
                )
                .result()
                .w()?;
            cu_graph_e.assume_init()
        };

        Ok(Self {
            graph: cu_graph,
            exec: cu_graph_e,
            stream: *cu_stream,
            device: device.clone(),
            input,
            _marker: PhantomData,
            ran_graph: Cell::new(false),
        })
    }

    /// Execute the graph with the provided inputs.
    ///
    /// All inputs are copied, so it may be detrimental to performance if large
    /// inputs are provided.
    ///
    /// # Panics
    /// - The inputs provided here must match the inputs provided upon construction.
    /// - The inputs provided here must all be continuous
    pub fn replay(&self, input: T) -> candle_core::Result<()> {
        self.input.load_inputs_inplace(input, &self.device)?;
        unsafe {
            driver::sys::lib()
                .cuGraphLaunch(self.exec, self.stream)
                .result()
                .w()?
        }
        self.ran_graph.set(true);
        self.device.synchronize()?;
        Ok(())
    }

    /// Requires that you have installed the [graphviz](https://graphviz.org/download/) library.
    /// Writes the graph to the specified path.
    pub fn output_dot<P: AsRef<Path>>(
        &self,
        out: P,
        format: GraphDumpFormat,
        verbosity: GraphDumpVerbosity,
    ) -> candle_core::Result<()> {
        let tmp = if let GraphDumpFormat::Dot = format {
            out.as_ref().to_string_lossy().trim().to_string()
        } else {
            format!("{}.dot", out.as_ref().display())
        };
        let cstr = unsafe { CString::from_vec_unchecked(tmp.as_bytes().to_vec()) };
        let verbosity = match verbosity {
            GraphDumpVerbosity::Verbose => {
                CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE as u32
            }
            GraphDumpVerbosity::Clean => 0,
        };
        unsafe { driver::sys::lib().cuGraphDebugDotPrint(self.graph, cstr.as_ptr(), verbosity) }
            .result()
            .w()?;
        let ty = match format {
            GraphDumpFormat::Png => "png",
            GraphDumpFormat::Svg => "svg",
            GraphDumpFormat::Dot => return Ok(()),
        };
        let command = Command::new("dot")
            .arg(format!("-T{ty}"))
            .arg(tmp)
            .output()
            .context("the graphviz utility must be installed: https://graphviz.org/download/")
            .map_err(candle_core::Error::debug)?
            .stdout;
        std::fs::write(out, command)?;
        Ok(())
    }

    /// Retrieve the nodes for this graph. Node dependency information is not tracked.
    pub fn nodes(&self) -> candle_core::Result<Vec<Node<'_>>> {
        println!("Getting nodes");
        let mut num_nodes = unsafe {
            let mut num_nodes = MaybeUninit::uninit();
            driver::sys::lib()
                .cuGraphGetNodes(self.graph, ptr::null_mut(), num_nodes.as_mut_ptr())
                .result()
                .w()?;
            num_nodes.assume_init()
        };
        let node_ptrs = unsafe {
            let mut nodes: Vec<CUgraphNode> = Vec::with_capacity(num_nodes);
            driver::sys::lib()
                .cuGraphGetNodes(self.graph, nodes.as_mut_ptr(), &mut num_nodes as *mut _)
                .result()
                .w()?;
            nodes.set_len(num_nodes);
            nodes
        };

        let mut nodes = Vec::new();
        for node in &node_ptrs {
            let node_type = unsafe {
                let mut node_type = MaybeUninit::uninit();
                driver::sys::lib()
                    .cuGraphNodeGetType(*node, node_type.as_mut_ptr())
                    .result()
                    .w()?;
                node_type.assume_init()
            };
            #[allow(clippy::single_match)]
            let data = match node_type {
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => {
                    let node_params = unsafe {
                        let mut node_params = MaybeUninit::uninit();
                        driver::sys::lib()
                            .cuGraphKernelNodeGetParams_v2(*node, node_params.as_mut_ptr())
                            .result()
                            .w()?;
                        node_params.assume_init()
                    };
                    let params = KernelLaunchParams {
                        grid_dim_x: node_params.gridDimX,
                        grid_dim_y: node_params.gridDimY,
                        grid_dim_z: node_params.gridDimZ,
                        block_dim_x: node_params.blockDimX,
                        block_dim_y: node_params.blockDimY,
                        block_dim_z: node_params.blockDimZ,
                        shared_mem_bytes: node_params.sharedMemBytes,
                    };
                    NodeData::Kernel {
                        launch_params: params,
                    }
                }
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_BATCH_MEM_OP => {
                    NodeData::BatchMemOp
                }
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL => {
                    NodeData::Conditional
                }
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY => NodeData::Empty,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EVENT_RECORD => {
                    NodeData::EventRecord
                }
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL => {
                    NodeData::ExtSemasSignal
                }
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT => {
                    NodeData::ExtSemasWait
                }
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_GRAPH => NodeData::Graph,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_HOST => NodeData::Host,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => NodeData::Memcpy,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => NodeData::Memset,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => NodeData::MemAlloc,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => NodeData::MemFree,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_WAIT_EVENT => NodeData::WaitEvent,
            };

            let node = Node {
                data,
                inner: *node,
                _marker: PhantomData,
            };
            nodes.push(node);
        }

        Ok(nodes)
    }
}

impl<T: GraphInput> Drop for Graph<T> {
    fn drop(&mut self) {
        if !self.ran_graph.get() {
            unsafe {
                driver::sys::lib()
                    .cuGraphLaunch(self.exec, self.stream)
                    .result()
                    .expect("Graph was not run, final run failed")
            }
            self.device
                .synchronize()
                .expect("Graph was not run, device sync failed")
        }
        unsafe { driver::sys::lib().cuGraphDestroy(self.graph) }
            .result()
            .expect("Graph destroy failed");
        unsafe { driver::sys::lib().cuGraphExecDestroy(self.exec) }
            .result()
            .expect("Graph destroy failed");
    }
}
