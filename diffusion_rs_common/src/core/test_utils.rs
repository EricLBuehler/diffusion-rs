use crate::core::{Result, Tensor};

#[macro_export]
macro_rules! test_device {
    // TODO: Switch to generating the two last arguments automatically once concat_idents is
    // stable. https://github.com/rust-lang/rust/issues/29599
    ($fn_name: ident, $test_cpu: ident, $test_cuda: ident, $test_metal: ident) => {
        #[test]
        fn $test_cpu() -> Result<()> {
            $fn_name(&Device::Cpu)
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn $test_cuda() -> Result<()> {
            $fn_name(&Device::new_cuda(0)?)
        }

        #[cfg(feature = "metal")]
        #[test]
        fn $test_metal() -> Result<()> {
            $fn_name(&Device::new_metal(0)?)
        }
    };
}

pub fn to_vec0_round(t: &Tensor, digits: i32) -> Result<f32> {
    let b = 10f32.powi(digits);
    let t = t.to_vec0::<f32>()?;
    Ok(f32::round(t * b) / b)
}

pub fn to_vec1_round(t: &Tensor, digits: i32) -> Result<Vec<f32>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec1::<f32>()?;
    let t = t.iter().map(|t| f32::round(t * b) / b).collect();
    Ok(t)
}

pub fn to_vec2_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec2::<f32>()?;
    let t = t
        .iter()
        .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
        .collect();
    Ok(t)
}

pub fn to_vec3_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}
