use crate::tensor::NdTensor;

/// Concatenate tensors along the given axis.
pub fn concat(tensors: &[&NdTensor<f32>], axis: usize) -> NdTensor<f32> {
    assert!(!tensors.is_empty(), "concat: empty tensor list");
    let ndim = tensors[0].ndim();
    assert!(axis < ndim, "concat: axis out of bounds");

    // Validate shapes match on all non-concat axes
    for t in &tensors[1..] {
        assert_eq!(t.ndim(), ndim);
        for (d, (&a, &b)) in tensors[0].shape.iter().zip(t.shape.iter()).enumerate() {
            if d != axis {
                assert_eq!(a, b, "concat: shape mismatch at dim {d}");
            }
        }
    }

    let mut new_shape = tensors[0].shape.clone();
    new_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();

    let outer: usize = new_shape[..axis].iter().product();
    let inner: usize = new_shape[axis + 1..].iter().product();
    let total: usize = new_shape.iter().product();
    let mut data = vec![0.0f32; total];

    let total_axis = new_shape[axis];
    let mut axis_offset = 0;
    for t in tensors {
        let t_axis = t.shape[axis];
        for o in 0..outer {
            for a in 0..t_axis {
                let src_off = o * t_axis * inner + a * inner;
                let dst_off = o * total_axis * inner + (axis_offset + a) * inner;
                data[dst_off..dst_off + inner].copy_from_slice(&t.data[src_off..src_off + inner]);
            }
        }
        axis_offset += t_axis;
    }

    NdTensor::from_vec(data, &new_shape)
}

/// Slice a tensor along the given axis.
pub fn slice_tensor(
    t: &NdTensor<f32>,
    axis: usize,
    start: i64,
    end: i64,
    step: i64,
) -> NdTensor<f32> {
    assert!(axis < t.ndim());
    assert!(step != 0, "slice: step cannot be 0");

    let axis_size = t.shape[axis] as i64;
    let start = if start < 0 {
        (axis_size + start).max(0)
    } else {
        start.min(axis_size)
    };
    let end = if end < 0 {
        (axis_size + end).max(0)
    } else {
        end.min(axis_size)
    };

    let indices: Vec<usize> = if step > 0 {
        (start..end)
            .step_by(step as usize)
            .map(|i| i as usize)
            .collect()
    } else {
        let mut v = Vec::new();
        let mut i = start;
        while i > end {
            v.push(i as usize);
            i += step;
        }
        v
    };

    let mut new_shape = t.shape.clone();
    new_shape[axis] = indices.len();

    let outer: usize = t.shape[..axis].iter().product();
    let inner: usize = t.shape[axis + 1..].iter().product();
    let total: usize = new_shape.iter().product();
    let mut data = vec![0.0f32; total];

    let orig_axis = t.shape[axis];
    for o in 0..outer {
        for (new_a, &orig_a) in indices.iter().enumerate() {
            let src_off = o * orig_axis * inner + orig_a * inner;
            let dst_off = o * indices.len() * inner + new_a * inner;
            data[dst_off..dst_off + inner].copy_from_slice(&t.data[src_off..src_off + inner]);
        }
    }

    NdTensor::from_vec(data, &new_shape)
}

/// Squeeze: remove dimensions of size 1 at the given axes.
pub fn squeeze(t: &mut NdTensor<f32>, axes: &[usize]) {
    let mut new_shape: Vec<usize> = Vec::new();
    for (i, &dim) in t.shape.iter().enumerate() {
        if axes.contains(&i) {
            assert_eq!(dim, 1, "squeeze: dim {i} is not 1");
        } else {
            new_shape.push(dim);
        }
    }
    if new_shape.is_empty() {
        new_shape.push(1);
    }
    t.reshape(&new_shape);
}

/// Unsqueeze: insert dimensions of size 1 at the given axes.
/// Axes refer to positions in the output shape.
pub fn unsqueeze(t: &mut NdTensor<f32>, axes: &[usize]) {
    let new_ndim = t.ndim() + axes.len();
    let mut new_shape = Vec::with_capacity(new_ndim);
    let mut src = 0;
    for i in 0..new_ndim {
        if axes.contains(&i) {
            new_shape.push(1);
        } else {
            new_shape.push(t.shape[src]);
            src += 1;
        }
    }
    t.reshape(&new_shape);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat() {
        let a = NdTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = NdTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = concat(&[&a, &b], 0);
        assert_eq!(c.shape, vec![4, 2]);
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let d = concat(&[&a, &b], 1);
        assert_eq!(d.shape, vec![2, 4]);
        assert_eq!(d.data, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_slice_basic() {
        let t = NdTensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], &[5]);
        let s = slice_tensor(&t, 0, 1, 4, 1);
        assert_eq!(s.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_slice_step() {
        let t = NdTensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[6]);
        let s = slice_tensor(&t, 0, 0, 6, 2);
        assert_eq!(s.data, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_squeeze() {
        let mut t = NdTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3, 1]);
        squeeze(&mut t, &[0, 2]);
        assert_eq!(t.shape, vec![3]);
    }

    #[test]
    fn test_unsqueeze() {
        let mut t = NdTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        unsqueeze(&mut t, &[0, 2]);
        assert_eq!(t.shape, vec![1, 3, 1]);
    }
}
