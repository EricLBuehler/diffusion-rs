use candle_core::{quantized::QTensor, DType, Device, Tensor};
use candle_nn::VarBuilder;
use diffusersrs_quant::{linear_no_bias, QuantizedConfig};

#[test]
fn test_load() -> candle_core::Result<()> {
    let dev = Device::Cpu;

    let vb = VarBuilder::from_tensors(
        candle_core::safetensors::load("tests/model.safetensors", &dev)?,
        DType::F32,
        &dev,
    );

    let truth = candle_core::safetensors::load("tests/model-00001-of-00002.safetensors", &dev)?
        ["model.layers.0.self_attn.q_proj.weight"]
        .clone()
        .to_dtype(DType::F32)?;

    let layer = linear_no_bias(
        1,
        1,
        Some(&QuantizedConfig::default()),
        vb.pp("model.layers.0.self_attn.q_proj"),
    )?;

    dbg!(&layer);

    // let xs = Tensor::randn(0f32, 1f32, (1, 8192), &dev)?;

    let res = layer.dequantize_w()?.to_dtype(DType::F32)?;
    dbg!(&res.mean_all()?);
    dbg!(&truth.mean_all()?);
    dbg!((res - truth.clone())?.abs()?.mean_all()?);

    let ggml_dequant =
        QTensor::quantize(&truth, candle_core::quantized::GgmlDType::Q4K)?.dequantize(&dev)?;
    dbg!((ggml_dequant - truth)?.abs()?.mean_all()?);

    Ok(())
}
