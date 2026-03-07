use crate::tensor::NdTensor;

/// Reshape tensor (zero-copy, just change shape/strides)
pub fn reshape(tensor: &mut NdTensor<f32>, new_shape: &[usize]) {
    tensor.reshape(new_shape);
}

/// Reshape with dynamic shape support.
/// - `u32::MAX` (0xFFFFFFFF) is treated as -1: inferred from total size.
/// - `0` copies the dimension from the input shape.
pub fn reshape_dynamic(tensor: &mut NdTensor<f32>, raw_shape: &[u32]) {
    let total = tensor.data.len();
    let mut new_shape: Vec<usize> = Vec::with_capacity(raw_shape.len());
    let mut infer_idx: Option<usize> = None;

    for (i, &dim) in raw_shape.iter().enumerate() {
        if dim == u32::MAX {
            // -1: infer
            assert!(
                infer_idx.is_none(),
                "reshape: only one -1 dimension allowed"
            );
            infer_idx = Some(i);
            new_shape.push(0); // placeholder
        } else if dim == 0 {
            // copy from input
            assert!(i < tensor.shape.len(), "reshape: dim 0 out of bounds");
            new_shape.push(tensor.shape[i]);
        } else {
            new_shape.push(dim as usize);
        }
    }

    if let Some(idx) = infer_idx {
        let known: usize = new_shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != idx)
            .map(|(_, &v)| v)
            .product();
        assert!(known > 0, "reshape: cannot infer with zero-sized dims");
        new_shape[idx] = total / known;
    }

    tensor.reshape(&new_shape);
}

/// Flatten from start_dim to end_dim (inclusive)
pub fn flatten(tensor: &mut NdTensor<f32>, start_dim: usize, end_dim: usize) {
    assert!(start_dim <= end_dim);
    assert!(end_dim < tensor.ndim());
    let mut new_shape = Vec::new();
    for i in 0..tensor.ndim() {
        if i == start_dim {
            let flat: usize = tensor.shape[start_dim..=end_dim].iter().product();
            new_shape.push(flat);
        } else if i > start_dim && i <= end_dim {
            continue;
        } else {
            new_shape.push(tensor.shape[i]);
        }
    }
    tensor.reshape(&new_shape);
}

/// Transpose: delegates to NdTensor::transpose
pub fn transpose(tensor: &NdTensor<f32>, dim0: usize, dim1: usize) -> NdTensor<f32> {
    tensor.transpose(dim0, dim1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten() {
        let mut t = NdTensor::from_vec((0..24).map(|x| x as f32).collect(), &[2, 3, 4]);
        flatten(&mut t, 1, 2);
        assert_eq!(t.shape, vec![2, 12]);
        assert_eq!(*t.get(&[0, 0]), 0.0);
        assert_eq!(*t.get(&[1, 0]), 12.0);
    }

    #[test]
    fn test_flatten_all() {
        let mut t = NdTensor::from_vec((0..24).map(|x| x as f32).collect(), &[2, 3, 4]);
        flatten(&mut t, 0, 2);
        assert_eq!(t.shape, vec![24]);
    }

    #[test]
    fn test_reshape_roundtrip() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut t = NdTensor::from_vec(data.clone(), &[2, 3, 4]);
        reshape(&mut t, &[4, 6]);
        assert_eq!(t.shape, vec![4, 6]);
        reshape(&mut t, &[2, 3, 4]);
        assert_eq!(t.shape, vec![2, 3, 4]);
        assert_eq!(t.data, data);
    }

    #[test]
    fn test_reshape_dynamic_infer() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut t = NdTensor::from_vec(data, &[2, 3, 4]);
        reshape_dynamic(&mut t, &[0, u32::MAX]); // [2, 12]
        assert_eq!(t.shape, vec![2, 12]);
    }

    #[test]
    fn test_reshape_dynamic_zero() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut t = NdTensor::from_vec(data, &[2, 3, 4]);
        reshape_dynamic(&mut t, &[0, 0, 4]); // [2, 3, 4] unchanged
        assert_eq!(t.shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_transpose_via_op() {
        let t = NdTensor::from_vec((0..6).map(|x| x as f32).collect(), &[2, 3]);
        let t2 = transpose(&t, 0, 1);
        assert_eq!(t2.shape, vec![3, 2]);
        assert_eq!(*t2.get(&[0, 1]), 3.0);
        assert_eq!(*t2.get(&[2, 0]), 2.0);
    }
}
