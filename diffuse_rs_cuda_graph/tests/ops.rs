use diffuse_rs_cuda_graph::{copy_inplace, Graph, GraphDumpFormat, GraphDumpVerbosity, GraphInput};
use half::bf16;

use std::f64::consts::E;

use diffuse_rs_common::core::{DType, Device, Tensor};

struct Inputs {
    x: Tensor,
}

impl GraphInput for Inputs {
    fn load_inputs_inplace(
        &self,
        input: Self,
        device: &Device,
    ) -> diffuse_rs_common::core::Result<()> {
        unsafe { copy_inplace(&input.x, &self.x, device)? };
        Ok(())
    }
}

#[test]
fn test_matmuls() -> anyhow::Result<()> {
    const N: usize = 50;
    const SHAPE: (usize, usize) = (4, 4);

    let device = Device::new_cuda_with_stream(0)?;

    let x = Tensor::ones(SHAPE, DType::BF16, &device)?;
    let mut y: Option<Tensor> = None;

    let graph = Graph::new(
        |input| {
            let x = &input.x;
            let out_data = x.log()?;
            y = Some(out_data);
            Ok(())
        },
        &device,
        Inputs { x },
    )?;

    graph.output_dot("out.png", GraphDumpFormat::Png, GraphDumpVerbosity::Verbose)?;

    for i in 1..=N {
        let new = Tensor::full(E.powi(i as i32), SHAPE, &device)?.to_dtype(DType::BF16)?;
        graph.replay(Inputs { x: new })?;
        if let Some(y) = &y {
            assert_eq!(
                y.to_vec2::<bf16>()?,
                Tensor::new(i as f32, &device)?
                    .to_dtype(DType::BF16)?
                    .broadcast_as(y.shape())?
                    .to_vec2::<bf16>()?
            );
        }
    }

    Ok(())
}
