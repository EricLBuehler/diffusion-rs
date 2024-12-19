use candle_core::{CpuStorage, Layout, Result, Shape, Tensor};

#[allow(dead_code)]
struct Sdpa {
    scale: f32,
    softcapping: f32,
}

impl candle_core::CustomOp3 for Sdpa {
    fn name(&self) -> &'static str {
        "metal-sdpa"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("SDPA has no cpu impl")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q: &candle_core::MetalStorage,
        q_l: &Layout,
        k: &candle_core::MetalStorage,
        k_l: &Layout,
        v: &candle_core::MetalStorage,
        v_l: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use crate::metal_kernels::SdpaDType;
        use candle_core::{backend::BackendStorage, DType, Shape, D};

        let device = q.device();

        let out_dims = vec![q_l.dim(0)?, q_l.dim(1)?, q_l.dim(2)?, v_l.dim(3)?];
        let elem_count: usize = out_dims.iter().product();

        let output = device.new_buffer(elem_count, q.dtype(), "sdpa_o")?;

        // q,k must have matching emb dim
        if q_l.dim(D::Minus1)? != k_l.dim(D::Minus1)? {
            candle_core::bail!("`q` and `k` last dims must match");
        }

        // k,v must have matching n kv heads
        if v_l.dim(D::Minus(3))? != k_l.dim(D::Minus(3))? {
            candle_core::bail!("`k` and `v` head dims must match");
        }

        // n_heads % n_kv_heads == 0; n_heads >= 1, n_kv_heads >= 1.
        if q_l.dim(D::Minus(3))? % k_l.dim(D::Minus(3))? != 0 {
            candle_core::bail!("query `n_heads` must be a multiple of `n_kv_heads`");
        }

        let k_head = k_l.dim(D::Minus1)?;
        let q_head = q_l.dim(D::Minus1)?;
        let q_seq = q_l.dim(2)?;

        let mut implementation_supports_use_case = q_head == k_head;
        let supported_head_dim =
            q_head == 32 || q_head == 64 || q_head == 96 || q_head == 128 || q_head == 256;

        const SDPA_FULL_THRESHOLD: usize = 2;

        let supports_sdpa_full =
            q_seq >= SDPA_FULL_THRESHOLD && supported_head_dim && q_head == k_head;
        let supports_sdpa_vector = q_seq == 1 && supported_head_dim;

        implementation_supports_use_case &= supports_sdpa_full || supports_sdpa_vector;

        if !supported_head_dim {
            candle_core::bail!(
                "Meta SDPA does not support q head dim {q_head}: q dims {:?}, k dims {:?}, v dims {:?}.",
                q_l.dims(),
                k_l.dims(),
                v_l.dims()
            );
        }
        if !implementation_supports_use_case {
            candle_core::bail!(
                "Meta SDPA does not support q dims {:?}, k dims {:?}, v dims {:?}.",
                q_l.dims(),
                k_l.dims(),
                v_l.dims()
            );
        }

        for t in [k.dtype(), v.dtype()] {
            if q.dtype() != t {
                candle_core::bail!("all q, k, v dtypes must match.");
            }
        }

        let itype = match q.dtype() {
            DType::BF16 => SdpaDType::BF16,
            DType::F16 => SdpaDType::F16,
            DType::F32 => SdpaDType::F32,
            other => candle_core::bail!("unsupported sdpa type {other:?}"),
        };

        let command_buffer = q.device().command_buffer()?;
        if supports_sdpa_vector {
            // Route to the 2 pass fused attention if the k seqlen is large.
            // https://github.com/ml-explore/mlx/pull/1597
            const TWO_PASS_K_THRESHOLD: usize = 1024;
            if k_l.dim(2)? >= TWO_PASS_K_THRESHOLD {
                let mut intermediate_shape = [
                    &out_dims[0..out_dims.len() - 2],
                    &[crate::metal_kernels::SDPA_2PASS_BLOCKS],
                    &[out_dims[out_dims.len() - 1]],
                ]
                .concat();
                let intermediate = device.new_buffer(
                    intermediate_shape.iter().product::<usize>(),
                    DType::F32,
                    "sdpa_2pass_intermediate",
                )?;
                let _ = intermediate_shape.pop().unwrap();
                let sums = device.new_buffer(
                    intermediate_shape.iter().product::<usize>(),
                    DType::F32,
                    "sdpa_2pass_sums",
                )?;
                let maxs = device.new_buffer(
                    intermediate_shape.iter().product::<usize>(),
                    DType::F32,
                    "sdpa_2pass_maxs",
                )?;

                command_buffer.set_label("vector_attention");
                crate::metal_kernels::call_sdpa_vector_2pass(
                    q.device().device(),
                    &command_buffer,
                    &crate::metal_kernels::Kernels::new(),
                    q_l.start_offset(),
                    q_l.dims(),
                    q.buffer(),
                    k_l.start_offset(),
                    k_l.dims(),
                    k_l.stride(),
                    k.buffer(),
                    v_l.start_offset(),
                    v_l.stride(),
                    v.buffer(),
                    &output,
                    &intermediate,
                    &sums,
                    &maxs,
                    self.scale,
                    self.softcapping,
                    itype,
                )
                .map_err(candle_core::Error::wrap)?;
            } else {
                command_buffer.set_label("vector_attention");
                crate::metal_kernels::call_sdpa_vector(
                    q.device().device(),
                    &command_buffer,
                    &crate::metal_kernels::Kernels::new(),
                    q_l.start_offset(),
                    q_l.dims(),
                    q.buffer(),
                    k_l.start_offset(),
                    k_l.dims(),
                    k_l.stride(),
                    k.buffer(),
                    v_l.start_offset(),
                    v_l.stride(),
                    v.buffer(),
                    &output,
                    self.scale,
                    self.softcapping,
                    itype,
                )
                .map_err(candle_core::Error::wrap)?;
            }
        } else if supports_sdpa_full {
            if q_l.dim(2)? != k_l.dim(2)? {
                candle_core::bail!(
                    "query and key sequence length must be equal if using full metal sdpa"
                )
            }

            command_buffer.set_label("full_attention");
            crate::metal_kernels::call_sdpa_full(
                q.device().device(),
                &command_buffer,
                &crate::metal_kernels::Kernels::new(),
                q_l.start_offset(),
                q_l.dims(),
                q.buffer(),
                k_l.start_offset(),
                k.buffer(),
                v_l.start_offset(),
                v.buffer(),
                &output,
                self.scale,
                self.softcapping,
                itype,
            )
            .map_err(candle_core::Error::wrap)?;
        } else {
            candle_core::bail!("must be vector or full sdpa kernel");
        }

        let newstorage =
            candle_core::MetalStorage::new(output, device.clone(), elem_count, q.dtype());
        Ok((newstorage, Shape::from_dims(&out_dims)))
    }
}

/// Scaled dot product attention with a fused kernel.
///
/// Computes softmax(qk^T*scale)v.
///
/// **Inputs shapes:**
/// - `q`: (bs, qhead, seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, v_hidden)
/// - `scale` is applied before softmax.
/// - If `softcapping` != 1.0:
///      - Computation is: softmax(tanh(qk^T*scale/cap)*cap)v
///
/// **Output shape:** (bs, qhead, seq, v_hidden)
///
/// **Supported head dims:** 32, 64, 96, 128, 256.
///
/// ## On Metal:
/// - If `seq` == 1:
///     - Use a vectorized kernel
///     - Supports `seq` != `kv_seq` (cross attn. support)
///     - Supports GQA when `qhead` is a multiple of `kv_head`
/// - Otherwise:
///     - Use an alternate kernel
///     - Requires `seq` == `kv_seq`
///     - GQA is not supported (requires `qhead` == `kv_head`)
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, softcapping: f32) -> Result<Tensor> {
    // Only use kernel for Metal as we only have one for that
    if q.device().is_metal() {
        q.apply_op3_no_bwd(k, v, &Sdpa { scale, softcapping })
    } else {
        let mut att = (q.matmul(&k.t()?)? * (scale as f64))?;
        if softcapping != 1.0 {
            att = (att / softcapping as f64)?;
            att = att.tanh()?;
            att = (att * softcapping as f64)?;
        }

        att = candle_nn::ops::softmax_last_dim(&att)?;
        att.matmul(v)
    }
}
