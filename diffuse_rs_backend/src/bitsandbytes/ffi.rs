use diffuse_rs_common::core::cuda::cudarc::driver::sys::CUstream;
use half::{bf16, f16};

#[allow(dead_code)]
extern "C" {
    pub(crate) fn dequantize_blockwise_f32_int8(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut f32,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );
    pub(crate) fn dequantize_blockwise_f32_fp4(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut f32,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );
    pub(crate) fn dequantize_blockwise_f32_nf4(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut f32,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );

    pub(crate) fn dequantize_blockwise_f16_int8(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut f16,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );
    pub(crate) fn dequantize_blockwise_f16_fp4(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut f16,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );
    pub(crate) fn dequantize_blockwise_f16_nf4(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut f16,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );

    pub(crate) fn dequantize_blockwise_bf16_int8(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut bf16,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );
    pub(crate) fn dequantize_blockwise_bf16_fp4(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut bf16,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );
    pub(crate) fn dequantize_blockwise_bf16_nf4(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut bf16,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );

    pub(crate) fn dequantize_8bit_kernel_f32(
        weight: *const i8,
        scb: *const f32,
        out: *mut f32,
        row: i32,
        col: i32,
        n: i32,
    );
    pub(crate) fn dequantize_8bit_kernel_f16(
        weight: *const i8,
        scb: *const f32,
        out: *mut f16,
        row: i32,
        col: i32,
        n: i32,
    );
    pub(crate) fn dequantize_8bit_kernel_bf16(
        weight: *const i8,
        scb: *const f32,
        out: *mut bf16,
        row: i32,
        col: i32,
        n: i32,
    );
}
