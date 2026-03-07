use crate::tensor::NdTensor;

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid_inplace(t: &mut NdTensor<f32>) {
    for v in t.data.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Numerically stable softmax along the given axis.
pub fn softmax(t: &NdTensor<f32>, axis: usize) -> NdTensor<f32> {
    assert!(axis < t.ndim(), "softmax: axis out of bounds");
    let mut out = t.clone();
    let axis_size = t.shape[axis];
    let outer: usize = t.shape[..axis].iter().product();
    let inner: usize = t.shape[axis + 1..].iter().product();

    for o in 0..outer {
        for i in 0..inner {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + i;
                max_val = max_val.max(t.data[idx]);
            }
            // Compute exp and sum
            let mut sum = 0.0f32;
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + i;
                let e = (t.data[idx] - max_val).exp();
                out.data[idx] = e;
                sum += e;
            }
            // Normalize
            let inv_sum = 1.0 / sum;
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + i;
                out.data[idx] *= inv_sum;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let mut t = NdTensor::from_vec(vec![0.0, 1.0, -1.0], &[3]);
        sigmoid_inplace(&mut t);
        assert!((t.data[0] - 0.5).abs() < 1e-6);
        assert!((t.data[1] - 0.7310586).abs() < 1e-5);
        assert!((t.data[2] - 0.2689414).abs() < 1e-5);
    }

    #[test]
    fn test_softmax() {
        let t = NdTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        let s = softmax(&t, 1);
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Values should be increasing
        assert!(s.data[0] < s.data[1]);
        assert!(s.data[1] < s.data[2]);
    }
}
