use rayon::prelude::*;
use wide::f32x8;

use crate::tensor::NdTensor;

/// Matrix multiplication with broadcasting support.
/// Supports 2D x 2D, 3D x 3D (batched), 3D x 2D (broadcast).
pub fn matmul(a: &NdTensor<f32>, b: &NdTensor<f32>) -> NdTensor<f32> {
    match (a.ndim(), b.ndim()) {
        (2, 2) => matmul_2d(a, b),
        (3, 3) => matmul_batched(a, b),
        (3, 2) => {
            let batch = a.shape[0];
            let m = a.shape[1];
            let k = a.shape[2];
            let n = b.shape[1];
            assert_eq!(b.shape[0], k, "matmul: inner dims mismatch");
            let mut out = NdTensor::zeros(&[batch, m, n]);
            for bi in 0..batch {
                let a_off = bi * m * k;
                let o_off = bi * m * n;
                gemm(
                    &a.data[a_off..a_off + m * k],
                    &b.data,
                    &mut out.data[o_off..o_off + m * n],
                    k,
                    n,
                );
            }
            out
        }
        (4, 4) => matmul_4d(a, b),
        _ => panic!("matmul: unsupported shapes {:?} x {:?}", a.shape, b.shape),
    }
}

fn matmul_2d(a: &NdTensor<f32>, b: &NdTensor<f32>) -> NdTensor<f32> {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    assert_eq!(b.shape[0], k, "matmul: inner dims mismatch");
    let mut out = NdTensor::zeros(&[m, n]);
    gemm(&a.data, &b.data, &mut out.data, k, n);
    out
}

fn matmul_batched(a: &NdTensor<f32>, b: &NdTensor<f32>) -> NdTensor<f32> {
    let batch = a.shape[0];
    assert_eq!(b.shape[0], batch, "matmul: batch dims mismatch");
    let m = a.shape[1];
    let k = a.shape[2];
    let n = b.shape[2];
    assert_eq!(b.shape[1], k, "matmul: inner dims mismatch");
    let mut out = NdTensor::zeros(&[batch, m, n]);
    for bi in 0..batch {
        let a_off = bi * m * k;
        let b_off = bi * k * n;
        let o_off = bi * m * n;
        gemm(
            &a.data[a_off..a_off + m * k],
            &b.data[b_off..b_off + k * n],
            &mut out.data[o_off..o_off + m * n],
            k,
            n,
        );
    }
    out
}

fn matmul_4d(a: &NdTensor<f32>, b: &NdTensor<f32>) -> NdTensor<f32> {
    let (b0, b1) = (a.shape[0], a.shape[1]);
    assert_eq!(b.shape[0], b0, "matmul 4d: dim0 mismatch");
    assert_eq!(b.shape[1], b1, "matmul 4d: dim1 mismatch");
    let m = a.shape[2];
    let k = a.shape[3];
    let n = b.shape[3];
    assert_eq!(b.shape[2], k, "matmul 4d: inner dims mismatch");
    let mut out = NdTensor::zeros(&[b0, b1, m, n]);
    for i0 in 0..b0 {
        for i1 in 0..b1 {
            let a_off = (i0 * b1 + i1) * m * k;
            let b_off = (i0 * b1 + i1) * k * n;
            let o_off = (i0 * b1 + i1) * m * n;
            gemm(
                &a.data[a_off..a_off + m * k],
                &b.data[b_off..b_off + k * n],
                &mut out.data[o_off..o_off + m * n],
                k,
                n,
            );
        }
    }
    out
}

/// General matrix multiply: C = A * B
/// A: (m, k), B: (k, n), C: (m, n)
fn gemm(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
    /// Columns handled per accumulator block. 64 floats keep the accumulators in
    /// registers while `b` is read as whole cache lines.
    const JB: usize = 64;

    // Loop order is what matters: reading a column of `b` one element at a time (stride
    // `n`) misses cache on every access, so instead broadcast one element of `a` and walk
    // a *row* of `b` contiguously, keeping the partial sums in vector accumulators.
    c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        let a_row = &a[i * k..(i + 1) * k];

        let mut j0 = 0;
        while j0 + JB <= n {
            let mut acc = [f32x8::ZERO; JB / 8];
            for (kk, &av) in a_row.iter().enumerate() {
                let bv = f32x8::splat(av);
                let b_row = &b[kk * n + j0..kk * n + j0 + JB];
                for (v, slot) in acc.iter_mut().enumerate() {
                    *slot += bv * f32x8::from(&b_row[v * 8..v * 8 + 8]);
                }
            }
            for (v, slot) in acc.iter().enumerate() {
                let arr: [f32; 8] = (*slot).into();
                c_row[j0 + v * 8..j0 + v * 8 + 8].copy_from_slice(&arr);
            }
            j0 += JB;
        }

        while j0 + 8 <= n {
            let mut acc = f32x8::ZERO;
            for (kk, &av) in a_row.iter().enumerate() {
                acc += f32x8::splat(av) * f32x8::from(&b[kk * n + j0..kk * n + j0 + 8]);
            }
            let arr: [f32; 8] = acc.into();
            c_row[j0..j0 + 8].copy_from_slice(&arr);
            j0 += 8;
        }
        while j0 < n {
            let mut sum = 0.0f32;
            for (kk, &av) in a_row.iter().enumerate() {
                sum += av * b[kk * n + j0];
            }
            c_row[j0] = sum;
            j0 += 1;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2d() {
        // (2,3) x (3,2) = (2,2)
        let a = NdTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = NdTensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape, vec![2, 2]);
        assert!((c.data[0] - 58.0).abs() < 1e-4); // 1*7+2*9+3*11
        assert!((c.data[1] - 64.0).abs() < 1e-4); // 1*8+2*10+3*12
        assert!((c.data[2] - 139.0).abs() < 1e-4); // 4*7+5*9+6*11
        assert!((c.data[3] - 154.0).abs() < 1e-4); // 4*8+5*10+6*12
    }

    #[test]
    fn test_matmul_3d_x_2d() {
        let a = NdTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2]);
        let b = NdTensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], &[2, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape, vec![2, 2, 2]);
        // batch 0: identity * b = b
        assert!((c.data[0] - 3.0).abs() < 1e-4);
        assert!((c.data[1] - 4.0).abs() < 1e-4);
        // batch 1: 2*identity * b = 2*b
        assert!((c.data[4] - 6.0).abs() < 1e-4);
        assert!((c.data[5] - 8.0).abs() < 1e-4);
    }
}
