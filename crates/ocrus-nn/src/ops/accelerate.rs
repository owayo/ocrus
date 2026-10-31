//! Optional system BLAS for macOS. The graph, model and preprocessing stay in Rust.
use std::ffi::c_int;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

/// Row-major C = A * B. Fall back to Rust when BLAS dimensions cannot represent it.
pub(super) fn multiply(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) -> bool {
    if n == 0 {
        return c.is_empty();
    }
    let m = c.len() / n;
    assert_eq!(
        c.len(),
        m.checked_mul(n).expect("GEMM output size overflow")
    );
    assert_eq!(a.len(), m.checked_mul(k).expect("GEMM left size overflow"));
    assert_eq!(b.len(), k.checked_mul(n).expect("GEMM right size overflow"));
    let (Ok(m), Ok(n), Ok(k)) = (c_int::try_from(m), c_int::try_from(n), c_int::try_from(k)) else {
        return false;
    };
    if m == 0 || k == 0 {
        c.fill(0.0);
        return true;
    }
    // SAFETY: checked, contiguous row-major slices cover all accesses. Rust's
    // exclusive output borrow prevents aliasing. Beta=0 replaces each output.
    unsafe {
        cblas_sgemm(
            101,
            111,
            111,
            m,
            n,
            k,
            1.0,
            a.as_ptr(),
            k,
            b.as_ptr(),
            n,
            0.0,
            c.as_mut_ptr(),
            n,
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangular_products_match_scalar_including_kernel_remainders() {
        for (m, k, n) in [(1, 1, 1), (3, 5, 7), (17, 13, 19), (31, 64, 33), (5, 0, 7)] {
            let a: Vec<_> = (0..m * k).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<_> = (0..k * n).map(|i| (i as f32 * 0.19).cos()).collect();
            let mut c = vec![f32::NAN; m * n];
            assert!(multiply(&a, &b, &mut c, k, n));
            for i in 0..m {
                for j in 0..n {
                    let expected: f32 = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
                    assert!(
                        (c[i * n + j] - expected).abs() < 1e-4,
                        "{m}x{k}x{n}, {i},{j}"
                    );
                }
            }
        }
    }
}
