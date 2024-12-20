use std::{marker::PhantomData, mem::MaybeUninit, ops::Deref};

use candle_core::cuda::cudarc::driver::{self, sys::CUgraphNode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelLaunchParams {
    pub grid_dim_x: u32,
    pub grid_dim_y: u32,
    pub grid_dim_z: u32,
    pub block_dim_x: u32,
    pub block_dim_y: u32,
    pub block_dim_z: u32,
    pub shared_mem_bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeData {
    Kernel { launch_params: KernelLaunchParams },
    Memcpy,
    Memset,
    Host,
    Graph,
    Empty,
    WaitEvent,
    EventRecord,
    ExtSemasSignal,
    ExtSemasWait,
    MemAlloc,
    MemFree,
    BatchMemOp,
    Conditional,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Node<'a> {
    pub(crate) data: NodeData,
    // Node pointer, *raw*
    pub(crate) inner: CUgraphNode,
    // Lifetime of node pointer
    pub(crate) _marker: PhantomData<&'a ()>,
}

impl Deref for Node<'_> {
    type Target = NodeData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl AsRef<NodeData> for Node<'_> {
    fn as_ref(&self) -> &NodeData {
        &self.data
    }
}

impl Node<'_> {
    pub fn is_kernel(&self) -> bool {
        matches!(self.data, NodeData::Kernel { .. })
    }

    pub fn update_kernel_launch_params(
        &mut self,
        launch_params: KernelLaunchParams,
    ) -> anyhow::Result<()> {
        match &mut self.data {
            NodeData::Kernel { launch_params: tgt } => {
                *tgt = launch_params;
                let mut node_params = unsafe {
                    let mut node_params = MaybeUninit::uninit();
                    driver::sys::lib()
                        .cuGraphKernelNodeGetParams_v2(self.inner, node_params.as_mut_ptr())
                        .result()?;
                    node_params.assume_init()
                };
                node_params.gridDimX = tgt.grid_dim_x;
                node_params.gridDimY = tgt.grid_dim_y;
                node_params.gridDimZ = tgt.grid_dim_z;
                node_params.blockDimX = tgt.block_dim_x;
                node_params.blockDimY = tgt.block_dim_y;
                node_params.blockDimZ = tgt.block_dim_z;
                node_params.sharedMemBytes = tgt.shared_mem_bytes;
                unsafe {
                    driver::sys::lib()
                        .cuGraphKernelNodeSetParams_v2(self.inner, &node_params as *const _)
                        .result()?;
                }
            }
            _ => anyhow::bail!("This node is not a kernel node!"),
        }
        Ok(())
    }
}
