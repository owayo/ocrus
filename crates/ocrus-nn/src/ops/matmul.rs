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
                    m,
                    k,
                    n,
                );
            }
            out
        }
        _ => panic!("matmul: unsupported shapes {:?} x {:?}", a.shape, b.shape),
    }
}

fn matmul_2d(a: &NdTensor<f32>, b: &NdTensor<f32>) -> NdTensor<f32> {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    assert_eq!(b.shape[0], k, "matmul: inner dims mismatch");
    let mut out = NdTensor::zeros(&[m, n]);
    gemm(&a.data, &b.data, &mut out.data, m, k, n);
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
            m,
            k,
            n,
        );
    }
    out
}

/// General matrix multiply: C = A * B
/// A: (m, k), B: (k, n), C: (m, n)
fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        for j in 0..n {
            let chunks = k / 8;
            let mut acc = f32x8::ZERO;
            for ch in 0..chunks {
                let off = ch * 8;
                let av = f32x8::from(&a_row[off..off + 8]);
                // Gather b column values
                let bv = f32x8::from([
                    b[(off) * n + j],
                    b[(off + 1) * n + j],
                    b[(off + 2) * n + j],
                    b[(off + 3) * n + j],
                    b[(off + 4) * n + j],
                    b[(off + 5) * n + j],
                    b[(off + 6) * n + j],
                    b[(off + 7) * n + j],
                ]);
                acc += av * bv;
            }
            let arr: [f32; 8] = acc.into();
            let mut dot: f32 = arr.iter().sum();
            for ki in (chunks * 8)..k {
                dot += a_row[ki] * b[ki * n + j];
            }
            c_row[j] = dot;
        }
    }
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
