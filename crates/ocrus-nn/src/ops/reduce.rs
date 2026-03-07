use crate::tensor::NdTensor;

/// ReduceMean along the given axes, optionally keeping dimensions.
pub fn reduce_mean(t: &NdTensor<f32>, axes: &[usize], keepdims: bool) -> NdTensor<f32> {
    assert!(!axes.is_empty());
    for &a in axes {
        assert!(a < t.ndim(), "reduce_mean: axis {a} out of bounds");
    }

    // Compute output shape
    let mut out_shape: Vec<usize> = Vec::new();
    for (i, &dim) in t.shape.iter().enumerate() {
        if axes.contains(&i) {
            if keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(dim);
        }
    }
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let reduce_count: usize = axes.iter().map(|&a| t.shape[a]).product();
    let out_size: usize = out_shape.iter().product();
    let mut out_data = vec![0.0f32; out_size];

    // Iterate over all elements, accumulate into output
    let ndim = t.ndim();
    let mut idx = vec![0usize; ndim];
    for flat in 0..t.data.len() {
        // Compute multi-dim index
        let mut rem = flat;
        for d in 0..ndim {
            idx[d] = rem / t.strides[d];
            rem %= t.strides[d];
        }

        // Compute output flat index
        let mut out_idx = 0;
        let mut out_stride = 1;
        for d in (0..ndim).rev() {
            if !axes.contains(&d) {
                out_idx += idx[d] * out_stride;
                let dim = if axes.contains(&d) { 1 } else { t.shape[d] };
                // Find corresponding dim in out_shape
                out_stride *= dim;
            } else if keepdims {
                // skip, idx would be 0 in output
            }
        }

        out_data[out_idx] += t.data[flat];
    }

    // Divide by count
    let inv = 1.0 / reduce_count as f32;
    for v in out_data.iter_mut() {
        *v *= inv;
    }

    NdTensor::from_vec(out_data, &out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_mean_axis0() {
        let t = NdTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = reduce_mean(&t, &[0], false);
        assert_eq!(r.shape, vec![3]);
        assert!((r.data[0] - 2.5).abs() < 1e-6); // (1+4)/2
        assert!((r.data[1] - 3.5).abs() < 1e-6); // (2+5)/2
        assert!((r.data[2] - 4.5).abs() < 1e-6); // (3+6)/2
    }

    #[test]
    fn test_reduce_mean_keepdims() {
        let t = NdTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = reduce_mean(&t, &[1], true);
        assert_eq!(r.shape, vec![2, 1]);
        assert!((r.data[0] - 2.0).abs() < 1e-6); // (1+2+3)/3
        assert!((r.data[1] - 5.0).abs() < 1e-6); // (4+5+6)/3
    }
}
