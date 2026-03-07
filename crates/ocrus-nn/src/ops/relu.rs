use wide::f32x8;

use crate::tensor::NdTensor;

/// In-place ReLU: max(0, x)
pub fn relu_inplace(tensor: &mut NdTensor<f32>) {
    let data = &mut tensor.data;
    let chunks = data.len() / 8;
    let zero = f32x8::ZERO;

    for i in 0..chunks {
        let offset = i * 8;
        let v = f32x8::from(&data[offset..offset + 8]);
        let result = v.max(zero);
        let arr: [f32; 8] = result.into();
        data[offset..offset + 8].copy_from_slice(&arr);
    }
    for i in (chunks * 8)..data.len() {
        data[i] = data[i].max(0.0);
    }
}

/// In-place HardSwish: x * min(max(x + 3, 0), 6) / 6
pub fn hard_swish_inplace(tensor: &mut NdTensor<f32>) {
    let data = &mut tensor.data;
    let three = f32x8::splat(3.0);
    let six = f32x8::splat(6.0);
    let zero = f32x8::ZERO;
    let chunks = data.len() / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let x = f32x8::from(&data[offset..offset + 8]);
        let inner = (x + three).max(zero).min(six);
        let result = x * inner / six;
        let arr: [f32; 8] = result.into();
        data[offset..offset + 8].copy_from_slice(&arr);
    }
    for i in (chunks * 8)..data.len() {
        let x = data[i];
        data[i] = x * ((x + 3.0).max(0.0).min(6.0)) / 6.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_positive_negative() {
        let data = vec![-3.0, -1.0, 0.0, 1.0, 2.0, -0.5, 3.0, -2.0, 4.0, -4.0];
        let mut t = NdTensor::from_vec(data, &[10]);
        relu_inplace(&mut t);
        assert_eq!(
            t.data,
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]
        );
    }

    #[test]
    fn test_hard_swish() {
        // x = 0 → 0 * 3/6 = 0
        // x = 3 → 3 * 6/6 = 3
        // x = -3 → -3 * 0/6 = 0
        // x = -4 → -4 * 0/6 = 0 (clamped)
        // x = 6 → 6 * 6/6 = 6
        let data = vec![0.0, 3.0, -3.0, -4.0, 6.0];
        let mut t = NdTensor::from_vec(data, &[5]);
        hard_swish_inplace(&mut t);
        assert!((t.data[0] - 0.0).abs() < 1e-6);
        assert!((t.data[1] - 3.0).abs() < 1e-6);
        assert!((t.data[2] - 0.0).abs() < 1e-6);
        assert!((t.data[3] - 0.0).abs() < 1e-6);
        assert!((t.data[4] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_simd_boundary() {
        // 11 elements: 8 SIMD + 3 remainder
        let data = vec![
            -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0,
        ];
        let mut t = NdTensor::from_vec(data, &[11]);
        relu_inplace(&mut t);
        assert_eq!(
            t.data,
            vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0, 0.0, 10.0, 0.0]
        );
    }
}
