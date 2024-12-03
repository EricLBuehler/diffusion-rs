use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use diffusersrs_quant::{linear_no_bias, QuantizedConfig};

#[test]
fn test_load() -> candle_core::Result<()> {
    let dev = Device::Cpu;

    let vb = VarBuilder::from_tensors(
        candle_core::safetensors::load("tests/test.safetensors", &dev)?,
        DType::F32,
        &dev,
    );

    let layer = linear_no_bias(
        1,
        1,
        Some(&QuantizedConfig::default()),
        vb.pp("model.layers.0.mlp.down_proj"),
    )?;

    dbg!(&layer);

    Ok(())
}
