#[cfg(feature = "metal")]
mod metal_sdpa_tests {
    #[test]
    fn sdpa_full() -> crate::core::Result<()> {
        use crate::core::{DType, Device, Tensor};

        // Force seqlen = 100
        const BS: usize = 4;
        const R: usize = 4;
        const L: usize = 4;
        const DK: usize = 64;
        const H: usize = 3;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();

        let device = Device::new_metal(0)?;

        let q = Tensor::randn(0f32, 1f32, (BS, H, R, DK), &device)?;
        let k = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let v = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;

        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = diffuse_rs_common::nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };

        let sdpa_output = diffuse_rs_common::nn::ops::sdpa(&q, &k, &v, scale as f32, 1.)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.0005, "{}", error);

        Ok(())
    }

    #[test]
    fn sdpa_vector() -> crate::core::Result<()> {
        use crate::core::{DType, Device, Tensor};

        // Allow vectorized, seqlen = 1
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 1;
        const DK: usize = 64;
        const H: usize = 3;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();

        let device = Device::new_metal(0)?;

        let q = Tensor::randn(0f32, 1f32, (BS, H, R, DK), &device)?;
        let k = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let v = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;

        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = diffuse_rs_common::nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };

        let sdpa_output = diffuse_rs_common::nn::ops::sdpa(&q, &k, &v, scale as f32, 1.)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.0001, "{}", error);

        Ok(())
    }

    #[test]
    fn sdpa_vector_2pass() -> crate::core::Result<()> {
        use crate::core::{DType, Device, Tensor};

        // Allow vectorized, seqlen = 1 but kseqlen is long (long context)
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 2048;
        const DK: usize = 64;
        const H: usize = 3;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();

        let device = Device::new_metal(0)?;

        let q = Tensor::randn(0f32, 1f32, (BS, H, R, DK), &device)?;
        let k = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let v = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;

        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = diffuse_rs_common::nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };

        let sdpa_output = diffuse_rs_common::nn::ops::sdpa(&q, &k, &v, scale as f32, 1.)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.002, "{}", error);

        Ok(())
    }

    #[test]
    fn sdpa_full_softcapping() -> crate::core::Result<()> {
        use crate::core::{DType, Device, Tensor};
        use std::ops::{Div, Mul};

        // Allow vectorized, seqlen = 1
        const BS: usize = 4;
        const R: usize = 4;
        const L: usize = 4;
        const DK: usize = 64;
        const H: usize = 3;
        const SOFTCAP: f64 = 50.;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();

        let device = Device::new_metal(0)?;

        let q = Tensor::randn(0f32, 1f32, (BS, H, R, DK), &device)?;
        let k = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let v = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;

        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = diffuse_rs_common::nn::ops::softmax_last_dim(
                &att.to_dtype(DType::F32)?
                    .div(SOFTCAP)?
                    .tanh()?
                    .mul(SOFTCAP)?,
            )?
            .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };

        let sdpa_output = diffuse_rs_common::nn::ops::sdpa(&q, &k, &v, scale as f32, SOFTCAP as f32)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.0004, "{}", error);

        Ok(())
    }

    #[test]
    fn sdpa_vector_softcapping() -> crate::core::Result<()> {
        use crate::core::{DType, Device, Tensor};
        use std::ops::{Div, Mul};

        // Allow vectorized, seqlen = 1
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 1;
        const DK: usize = 64;
        const H: usize = 3;
        const SOFTCAP: f64 = 50.;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();

        let device = Device::new_metal(0)?;

        let q = Tensor::randn(0f32, 1f32, (BS, H, R, DK), &device)?;
        let k = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let v = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;

        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = diffuse_rs_common::nn::ops::softmax_last_dim(
                &att.to_dtype(DType::F32)?
                    .div(SOFTCAP)?
                    .tanh()?
                    .mul(SOFTCAP)?,
            )?
            .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };

        let sdpa_output = diffuse_rs_common::nn::ops::sdpa(&q, &k, &v, scale as f32, SOFTCAP as f32)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.0001, "{}", error);

        Ok(())
    }

    #[test]
    fn sdpa_vector_2pass_softcapping() -> crate::core::Result<()> {
        use crate::core::{DType, Device, Tensor};
        use std::ops::{Div, Mul};

        // Allow vectorized, seqlen = 1 but kseqlen is long (long context)
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 2048;
        const DK: usize = 64;
        const H: usize = 3;
        const SOFTCAP: f64 = 50.;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();

        let device = Device::new_metal(0)?;

        let q = Tensor::randn(0f32, 1f32, (BS, H, R, DK), &device)?;
        let k = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let v = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;

        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = diffuse_rs_common::nn::ops::softmax_last_dim(
                &att.to_dtype(DType::F32)?
                    .div(SOFTCAP)?
                    .tanh()?
                    .mul(SOFTCAP)?,
            )?
            .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };

        let sdpa_output = diffuse_rs_common::nn::ops::sdpa(&q, &k, &v, scale as f32, SOFTCAP as f32)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.0021, "{}", error);

        Ok(())
    }

    #[test]
    fn sdpa_vector_cross() -> crate::core::Result<()> {
        use crate::core::{DType, Device, Tensor};

        // Allow vectorized, seqlen = 1. Simulat cross attention case where R != L, R = 1
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 24;
        const DK: usize = 64;
        const H: usize = 3;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();

        let device = Device::new_metal(0)?;

        let q = Tensor::randn(0f32, 1f32, (BS, H, R, DK), &device)?;
        let k = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let v = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;

        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = diffuse_rs_common::nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };

        let sdpa_output = diffuse_rs_common::nn::ops::sdpa(&q, &k, &v, scale as f32, 1.)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.0017, "{}", error);

        Ok(())
    }

    #[test]
    fn attn_softmax_mask() -> crate::core::Result<()> {
        use crate::core::{Device, Tensor};

        let device = Device::new_metal(0)?;

        let tensor = Tensor::randn(0f32, 1f32, (4, 32, 64, 64), &device)?;
        let truemask = Tensor::full(f32::MIN, (64, 64), &device)?.contiguous()?;

        let ground_truth = diffuse_rs_common::nn::ops::softmax_last_dim(&tensor.broadcast_add(&truemask)?)?;

        let softmax_out = diffuse_rs_common::nn::ops::attn_softmax_last_dim(&tensor, &truemask, 1.)?;

        let error: f32 = ((&ground_truth - &softmax_out)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error < 1e-5);

        Ok(())
    }

    #[test]
    fn attn_softmax_mask_scale() -> crate::core::Result<()> {
        use crate::core::{DType, Device, Tensor};

        let device = Device::new_metal(0)?;

        let tensor = Tensor::randn(0f32, 1f32, (4, 32, 64, 64), &device)?.to_dtype(DType::BF16)?;
        let truemask = Tensor::full(half::bf16::MIN, (64, 64), &device)?
            .contiguous()?
            .to_dtype(DType::BF16)?;

        let scale = 0.1f32;

        let ground_truth =
            diffuse_rs_common::nn::ops::softmax_last_dim(&(tensor.broadcast_add(&truemask)? * scale as f64)?)?
                .to_dtype(DType::F32)?;

        let softmax_out = diffuse_rs_common::nn::ops::attn_softmax_last_dim(&tensor, &truemask, scale)?
            .to_dtype(DType::F32)?;

        let error: f32 = ((&ground_truth - &softmax_out)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_dtype(DType::F32)?
            .to_scalar()?;

        assert!(error < 1e-5, "{error}");

        Ok(())
    }

    #[test]
    fn attn_softmax_mask_novec() -> crate::core::Result<()> {
        use crate::core::{Device, Tensor};

        let device = Device::new_metal(0)?;

        let tensor = Tensor::randn(0f32, 1f32, (4, 32, 64, 63), &device)?;
        let truemask = Tensor::full(f32::MIN, (64, 63), &device)?.contiguous()?;

        let ground_truth = diffuse_rs_common::nn::ops::softmax_last_dim(&tensor.broadcast_add(&truemask)?)?;

        let softmax_out = diffuse_rs_common::nn::ops::attn_softmax_last_dim(&tensor, &truemask, 1.)?;

        let error: f32 = ((&ground_truth - &softmax_out)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error < 1e-5);

        Ok(())
    }

    #[test]
    fn attn_softmax_mask_scale_novec() -> crate::core::Result<()> {
        use crate::core::{DType, Device, Tensor};

        let device = Device::new_metal(0)?;

        let tensor = Tensor::randn(0f32, 1f32, (4, 32, 64, 63), &device)?.to_dtype(DType::BF16)?;
        let truemask = Tensor::full(half::bf16::MIN, (64, 63), &device)?
            .contiguous()?
            .to_dtype(DType::BF16)?;

        let scale = 0.1f32;

        let ground_truth =
            diffuse_rs_common::nn::ops::softmax_last_dim(&(tensor.broadcast_add(&truemask)? * scale as f64)?)?
                .to_dtype(DType::F32)?;

        let softmax_out = diffuse_rs_common::nn::ops::attn_softmax_last_dim(&tensor, &truemask, scale)?
            .to_dtype(DType::F32)?;

        let error: f32 = ((&ground_truth - &softmax_out)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_dtype(DType::F32)?
            .to_scalar()?;

        assert!(error < 1e-5, "{error}");

        Ok(())
    }
}
