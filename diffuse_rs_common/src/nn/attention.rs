use crate::core::{Result, Tensor};

/// Computes (softmax(QK^T*sqrt(d_k)) + M)V. `M` is the attention mask, and is a bias (0 for unmasked, -inf for masked).
///
/// The attention implementation is automatically accelerated and dispatched as follows:
/// 1) If `use_flash_attn == true`, use a Flash Attention V2 kernel
/// 2) Otherwise, use SDPA with fusion of softmax scale and attention bias application
///
/// Note that there may be minute differences in output because floating point operations are not associative.
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    seq_len: usize,
) -> Result<Tensor> {
    let att = match mask {
        Some(mask) => {
            let (b, n, s, _h) = q.dims4()?;
            let mut mask_and_output = mask.broadcast_as((b, n, s, s))?.contiguous()?;
            q.contiguous()?.matmul_with_alpha_beta(
                &k.t()?.contiguous()?,
                &mut mask_and_output,
                Some(scale),
            )?;
            mask_and_output
        }
        None => q
            .contiguous()?
            .matmul_with_alpha(&k.t()?.contiguous()?, Some(scale))?,
    };

    let att = crate::nn::ops::softmax_last_dim(&att)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    att.matmul(&v.contiguous()?)
}
