use crate::tensor::NdTensor;

/// Element-wise power with broadcasting.
pub fn pow_tensor(a: &NdTensor<f32>, b: &NdTensor<f32>) -> NdTensor<f32> {
    NdTensor::broadcast_binary_op(a, b, |x, y| x.powf(y))
}

/// Element-wise power with a scalar exponent (fast path).
pub fn pow_scalar(t: &NdTensor<f32>, exp: f32) -> NdTensor<f32> {
    let data: Vec<f32> = t.data.iter().map(|&x| x.powf(exp)).collect();
    NdTensor::from_vec(data, &t.shape)
}

/// In-place element-wise square root.
pub fn sqrt_inplace(t: &mut NdTensor<f32>) {
    for v in t.data.iter_mut() {
        *v = v.sqrt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow_scalar() {
        let t = NdTensor::from_vec(vec![1.0, 4.0, 9.0], &[3]);
        let r = pow_scalar(&t, 0.5);
        assert!((r.data[0] - 1.0).abs() < 1e-6);
        assert!((r.data[1] - 2.0).abs() < 1e-6);
        assert!((r.data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_pow_tensor() {
        let a = NdTensor::from_vec(vec![2.0, 3.0], &[2]);
        let b = NdTensor::from_vec(vec![3.0], &[1]);
        let r = pow_tensor(&a, &b);
        assert!((r.data[0] - 8.0).abs() < 1e-4);
        assert!((r.data[1] - 27.0).abs() < 1e-4);
    }

    #[test]
    fn test_sqrt() {
        let mut t = NdTensor::from_vec(vec![4.0, 9.0, 16.0], &[3]);
        sqrt_inplace(&mut t);
        assert!((t.data[0] - 2.0).abs() < 1e-6);
        assert!((t.data[1] - 3.0).abs() < 1e-6);
        assert!((t.data[2] - 4.0).abs() < 1e-6);
    }
}
