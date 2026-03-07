use std::fmt;

/// N-dimensional tensor with contiguous storage.
#[derive(Clone)]
pub struct NdTensor<T: Clone> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<T: Clone + Default> NdTensor<T> {
    /// Create a new tensor with the given shape, filled with default values.
    pub fn zeros(shape: &[usize]) -> Self {
        let strides = compute_strides(shape);
        let size: usize = shape.iter().product();
        Self {
            data: vec![T::default(); size],
            shape: shape.to_vec(),
            strides,
        }
    }

    /// Create a tensor from existing data and shape.
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "Data length {} != shape product {}",
            data.len(),
            expected
        );
        let strides = compute_strides(shape);
        Self {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get element at multi-dimensional index.
    pub fn get(&self, idx: &[usize]) -> &T {
        let offset = self.offset(idx);
        &self.data[offset]
    }

    /// Get mutable element at multi-dimensional index.
    pub fn get_mut(&mut self, idx: &[usize]) -> &mut T {
        let offset = self.offset(idx);
        &mut self.data[offset]
    }

    /// Compute linear offset from multi-dimensional index.
    fn offset(&self, idx: &[usize]) -> usize {
        assert_eq!(idx.len(), self.shape.len());
        idx.iter()
            .zip(self.strides.iter())
            .map(|(i, s)| i * s)
            .sum()
    }

    /// Reshape without copying data (must have same total elements).
    pub fn reshape(&mut self, new_shape: &[usize]) {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(self.data.len(), new_size, "Cannot reshape: size mismatch");
        self.shape = new_shape.to_vec();
        self.strides = compute_strides(new_shape);
    }

    /// Transpose dimensions (swap two axes).
    /// Note: This copies data to maintain contiguous layout.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let ndim = self.ndim();
        assert!(dim0 < ndim && dim1 < ndim);

        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);

        // Copy data in transposed order
        let total = self.data.len();
        let mut new_data = vec![T::default(); total];
        let out_strides = compute_strides(&new_shape);

        let mut src_idx = vec![0usize; ndim];
        for i in 0..total {
            // Convert flat index to multi-dim using original strides
            let mut remaining = i;
            for (d, idx) in src_idx.iter_mut().enumerate() {
                *idx = remaining / self.strides[d];
                remaining %= self.strides[d];
            }
            // Swap dimensions for destination
            let mut dst_idx = src_idx.clone();
            dst_idx.swap(dim0, dim1);
            let dst_offset: usize = dst_idx
                .iter()
                .zip(out_strides.iter())
                .map(|(a, b)| a * b)
                .sum();
            new_data[dst_offset] = self.data[i].clone();
        }

        Self {
            data: new_data,
            shape: new_shape,
            strides: out_strides,
        }
    }

    /// Transpose with arbitrary permutation.
    /// `perm` specifies the new axis order, e.g., [0,2,1,3].
    pub fn transpose_perm(&self, perm: &[usize]) -> Self {
        let ndim = self.ndim();
        assert_eq!(perm.len(), ndim, "perm length must match ndim");

        let mut new_shape = vec![0usize; ndim];
        for (i, &p) in perm.iter().enumerate() {
            new_shape[i] = self.shape[p];
        }

        let total = self.data.len();
        let mut new_data = vec![T::default(); total];
        let out_strides = compute_strides(&new_shape);

        let mut src_idx = vec![0usize; ndim];
        for i in 0..total {
            let mut remaining = i;
            for (d, idx) in src_idx.iter_mut().enumerate() {
                *idx = remaining / self.strides[d];
                remaining %= self.strides[d];
            }
            let mut dst_offset = 0;
            for (d, &p) in perm.iter().enumerate() {
                dst_offset += src_idx[p] * out_strides[d];
            }
            new_data[dst_offset] = self.data[i].clone();
        }

        Self {
            data: new_data,
            shape: new_shape,
            strides: out_strides,
        }
    }

    /// Get a slice of data for a specific batch/channel.
    /// Returns (data_slice, spatial_shape) for the remaining dimensions.
    pub fn slice_2d(&self, outer_indices: &[usize]) -> (&[T], &[usize]) {
        assert!(outer_indices.len() < self.ndim());
        let offset: usize = outer_indices
            .iter()
            .zip(self.strides.iter())
            .map(|(i, s)| i * s)
            .sum();
        let inner_size: usize = self.strides[outer_indices.len() - 1];
        (
            &self.data[offset..offset + inner_size],
            &self.shape[outer_indices.len()..],
        )
    }

    /// Get a mutable slice of data for a specific batch/channel.
    pub fn slice_2d_mut(&mut self, outer_indices: &[usize]) -> (&mut [T], Vec<usize>) {
        assert!(outer_indices.len() < self.ndim());
        let offset: usize = outer_indices
            .iter()
            .zip(self.strides.iter())
            .map(|(i, s)| i * s)
            .sum();
        let inner_size: usize = self.strides[outer_indices.len() - 1];
        let remaining_shape = self.shape[outer_indices.len()..].to_vec();
        (&mut self.data[offset..offset + inner_size], remaining_shape)
    }
}

impl NdTensor<f32> {
    /// Compute numpy-style broadcast shape.
    pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
        let max_ndim = a.len().max(b.len());
        let mut result = vec![0usize; max_ndim];
        for i in 0..max_ndim {
            let da = if i < max_ndim - a.len() {
                1
            } else {
                a[i - (max_ndim - a.len())]
            };
            let db = if i < max_ndim - b.len() {
                1
            } else {
                b[i - (max_ndim - b.len())]
            };
            assert!(
                da == db || da == 1 || db == 1,
                "broadcast: incompatible shapes {a:?} and {b:?} at dim {i}"
            );
            result[i] = da.max(db);
        }
        result
    }

    /// Apply a binary operation with numpy-style broadcasting.
    pub fn broadcast_binary_op(
        a: &NdTensor<f32>,
        b: &NdTensor<f32>,
        op: fn(f32, f32) -> f32,
    ) -> NdTensor<f32> {
        // Same-shape fast path
        if a.shape == b.shape {
            let data: Vec<f32> = a
                .data
                .iter()
                .zip(b.data.iter())
                .map(|(&x, &y)| op(x, y))
                .collect();
            return NdTensor::from_vec(data, &a.shape);
        }

        // Scalar fast paths
        if b.data.len() == 1 {
            let bv = b.data[0];
            let data: Vec<f32> = a.data.iter().map(|&x| op(x, bv)).collect();
            return NdTensor::from_vec(data, &a.shape);
        }
        if a.data.len() == 1 {
            let av = a.data[0];
            let data: Vec<f32> = b.data.iter().map(|&y| op(av, y)).collect();
            return NdTensor::from_vec(data, &b.shape);
        }

        // General broadcasting
        let out_shape = Self::broadcast_shape(&a.shape, &b.shape);
        let total: usize = out_shape.iter().product();
        let ndim = out_shape.len();
        let mut data = vec![0.0f32; total];

        // Precompute strides for output and padded input shapes
        let out_strides = compute_strides(&out_shape);

        let pad_a = pad_shape(&a.shape, ndim);
        let pad_b = pad_shape(&b.shape, ndim);
        let strides_a = compute_strides(&pad_a);
        let strides_b = compute_strides(&pad_b);

        let mut idx = vec![0usize; ndim];
        for (flat, out_val) in data.iter_mut().enumerate() {
            // Decompose flat index
            let mut rem = flat;
            for (d, slot) in idx.iter_mut().enumerate() {
                *slot = rem / out_strides[d];
                rem %= out_strides[d];
            }
            // Map to input indices (clamp for broadcast dims)
            let mut a_off = 0;
            let mut b_off = 0;
            for d in 0..ndim {
                let ai = if pad_a[d] == 1 { 0 } else { idx[d] };
                let bi = if pad_b[d] == 1 { 0 } else { idx[d] };
                a_off += ai * strides_a[d];
                b_off += bi * strides_b[d];
            }
            *out_val = op(a.data[a_off], b.data[b_off]);
        }

        NdTensor::from_vec(data, &out_shape)
    }
}

/// Pad shape to target ndim by prepending 1s.
fn pad_shape(shape: &[usize], target_ndim: usize) -> Vec<usize> {
    let mut padded = vec![1usize; target_ndim - shape.len()];
    padded.extend_from_slice(shape);
    padded
}

impl<T: Clone + Default + fmt::Debug> fmt::Debug for NdTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NdTensor")
            .field("shape", &self.shape)
            .field("len", &self.data.len())
            .finish()
    }
}

/// Compute row-major strides from shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t: NdTensor<f32> = NdTensor::zeros(&[2, 3, 4]);
        assert_eq!(t.len(), 24);
        assert_eq!(t.shape, vec![2, 3, 4]);
        assert_eq!(t.strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_from_vec() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let t = NdTensor::from_vec(data, &[3, 4]);
        assert_eq!(*t.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_reshape() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut t = NdTensor::from_vec(data, &[2, 3, 4]);
        t.reshape(&[6, 4]);
        assert_eq!(t.shape, vec![6, 4]);
        assert_eq!(*t.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_transpose() {
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let t = NdTensor::from_vec(data, &[2, 3]);
        let t2 = t.transpose(0, 1);
        assert_eq!(t2.shape, vec![3, 2]);
        assert_eq!(*t2.get(&[0, 0]), 0.0);
        assert_eq!(*t2.get(&[0, 1]), 3.0);
        assert_eq!(*t2.get(&[1, 0]), 1.0);
    }

    #[test]
    fn test_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        assert_eq!(compute_strides(&[2, 3]), vec![3, 1]);
    }
}
