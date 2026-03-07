use crate::tensor::NdTensor;

/// Fused Layer Normalization using Welford's online algorithm.
/// Normalizes along the last `ndim - axis` dimensions.
pub fn layer_norm(
    t: &NdTensor<f32>,
    gamma: &NdTensor<f32>,
    beta: &NdTensor<f32>,
    axis: usize,
    eps: f32,
) -> NdTensor<f32> {
    assert!(axis < t.ndim());
    let outer: usize = t.shape[..axis].iter().product();
    let inner: usize = t.shape[axis..].iter().product();

    assert_eq!(gamma.data.len(), inner);
    assert_eq!(beta.data.len(), inner);

    let mut out = t.clone();

    for o in 0..outer {
        let base = o * inner;
        let slice = &t.data[base..base + inner];

        // Welford's 1-pass mean/variance
        let mut mean = 0.0f64;
        let mut m2 = 0.0f64;
        for (i, &v) in slice.iter().enumerate() {
            let n = (i + 1) as f64;
            let delta = v as f64 - mean;
            mean += delta / n;
            let delta2 = v as f64 - mean;
            m2 += delta * delta2;
        }
        let variance = m2 / inner as f64;
        let inv_std = 1.0 / ((variance + eps as f64).sqrt());

        let out_slice = &mut out.data[base..base + inner];
        for (i, val) in out_slice.iter_mut().enumerate() {
            let normalized = (slice[i] as f64 - mean) * inv_std;
            *val = (normalized * gamma.data[i] as f64 + beta.data[i] as f64) as f32;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        // Simple 1D case: [1, 2, 3, 4] with gamma=1, beta=0
        let t = NdTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = NdTensor::from_vec(vec![1.0; 4], &[4]);
        let beta = NdTensor::from_vec(vec![0.0; 4], &[4]);
        let out = layer_norm(&t, &gamma, &beta, 1, 1e-5);
        assert_eq!(out.shape, vec![1, 4]);
        // Mean = 2.5, var = 1.25
        // Normalized: [-1.342, -0.447, 0.447, 1.342] approx
        let sum: f32 = out.data.iter().sum();
        assert!(sum.abs() < 1e-4, "mean should be ~0");
    }
}
