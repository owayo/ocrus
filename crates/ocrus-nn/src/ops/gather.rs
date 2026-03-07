use crate::tensor::NdTensor;

/// Gather elements along the given axis using float indices (cast to usize).
pub fn gather(t: &NdTensor<f32>, indices: &NdTensor<f32>, axis: usize) -> NdTensor<f32> {
    assert!(axis < t.ndim());

    // Output shape: replace t.shape[axis] with indices.shape
    // For simplicity, handle the common case: indices is 1D
    let ndim = t.ndim();
    let outer: usize = t.shape[..axis].iter().product();
    let axis_size = t.shape[axis];
    let inner: usize = t.shape[axis + 1..].iter().product();

    let num_indices = indices.data.len();
    let mut out_shape = t.shape[..axis].to_vec();
    out_shape.extend_from_slice(&indices.shape);
    out_shape.extend_from_slice(&t.shape[axis + 1..]);

    let out_size: usize = out_shape.iter().product();
    let mut data = vec![0.0f32; out_size];

    for o in 0..outer {
        for (ii, &idx_f) in indices.data.iter().enumerate() {
            let idx = idx_f as usize;
            assert!(
                idx < axis_size,
                "gather: index {idx} out of bounds for axis size {axis_size}"
            );
            let src_base = o * axis_size * inner + idx * inner;
            let dst_base = o * num_indices * inner + ii * inner;
            data[dst_base..dst_base + inner].copy_from_slice(&t.data[src_base..src_base + inner]);
        }
    }

    NdTensor::from_vec(data, &out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_axis0() {
        let t = NdTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let indices = NdTensor::from_vec(vec![0.0, 2.0], &[2]);
        let out = gather(&t, &indices, 0);
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.data, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_gather_axis1() {
        let t = NdTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let indices = NdTensor::from_vec(vec![0.0, 2.0], &[2]);
        let out = gather(&t, &indices, 1);
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.data, vec![1.0, 3.0, 4.0, 6.0]);
    }
}
