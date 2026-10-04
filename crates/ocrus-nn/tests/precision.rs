//! Numerical precision tests for the ocrus-nn kernels.
//!
//! These check that the SIMD paths agree with straightforward scalar code. Whole-graph
//! execution is covered by `ocnn_golden.rs`, which replays the outputs recorded by the
//! converter against the real model.

use ocrus_nn::ops::batchnorm::{BnParams, batchnorm_inplace};
use ocrus_nn::ops::conv2d::conv2d_pointwise;
use ocrus_nn::ops::linear::linear;
use ocrus_nn::ops::relu::{hard_swish_inplace, relu_inplace};
use ocrus_nn::tensor::NdTensor;

// ============================================================
// 1. SIMD vs scalar consistency tests
// ============================================================

#[test]
fn relu_simd_scalar_consistency() {
    // 17 elements: 2 SIMD chunks (16) + 1 scalar remainder
    let data: Vec<f32> = (-8..9).map(|x| x as f32 * 0.7).collect();
    let mut t = NdTensor::from_vec(data.clone(), &[17]);
    relu_inplace(&mut t);

    for (i, &orig) in data.iter().enumerate() {
        let expected = orig.max(0.0);
        assert!(
            (t.data[i] - expected).abs() < 1e-7,
            "relu mismatch at {i}: got {}, expected {expected}",
            t.data[i]
        );
    }
}

#[test]
fn hard_swish_simd_scalar_consistency() {
    // 19 elements: 2 SIMD chunks + 3 remainder
    let data: Vec<f32> = (-9..10).map(|x| x as f32 * 0.5).collect();
    let mut t = NdTensor::from_vec(data.clone(), &[19]);
    hard_swish_inplace(&mut t);

    for (i, &x) in data.iter().enumerate() {
        let expected = x * (x + 3.0).clamp(0.0, 6.0) / 6.0;
        assert!(
            (t.data[i] - expected).abs() < 1e-6,
            "hard_swish mismatch at {i}: got {}, expected {expected}",
            t.data[i]
        );
    }
}

#[test]
fn batchnorm_simd_scalar_consistency() {
    // 2 channels, 11 spatial elements (8 SIMD + 3 scalar)
    let data: Vec<f32> = (0..22).map(|x| x as f32 * 0.3 - 3.0).collect();
    let mut t = NdTensor::from_vec(data.clone(), &[1, 2, 1, 11]);
    let params = vec![
        BnParams {
            gamma: 2.0,
            beta: -1.0,
            running_mean: 0.5,
            running_var: 4.0,
            eps: 1e-5,
        },
        BnParams {
            gamma: 0.5,
            beta: 3.0,
            running_mean: -1.0,
            running_var: 2.0,
            eps: 1e-5,
        },
    ];
    batchnorm_inplace(&mut t, &params);

    for (ch, p) in params.iter().enumerate() {
        let scale = p.gamma / (p.running_var + p.eps).sqrt();
        let bias = p.beta - p.running_mean * scale;
        for i in 0..11 {
            let idx = ch * 11 + i;
            let expected = data[idx] * scale + bias;
            assert!(
                (t.data[idx] - expected).abs() < 1e-5,
                "batchnorm mismatch at ch={ch}, i={i}: got {}, expected {expected}",
                t.data[idx]
            );
        }
    }
}

#[test]
fn linear_simd_scalar_consistency() {
    // 13 input features: 1 SIMD chunk + 5 scalar remainder
    let in_data: Vec<f32> = (0..13).map(|x| x as f32 * 0.1).collect();
    let w_data: Vec<f32> = (0..26).map(|x| (x as f32 - 13.0) * 0.05).collect();
    let b_data = vec![1.0, -0.5];

    let input = NdTensor::from_vec(in_data.clone(), &[1, 13]);
    let weight = NdTensor::from_vec(w_data.clone(), &[2, 13]);
    let bias = NdTensor::from_vec(b_data.clone(), &[2]);
    let output = linear(&input, &weight, Some(&bias));

    for o in 0..2 {
        let mut expected = b_data[o];
        for i in 0..13 {
            expected += in_data[i] * w_data[o * 13 + i];
        }
        assert!(
            (output.data[o] - expected).abs() < 1e-4,
            "linear mismatch at {o}: got {}, expected {expected}",
            output.data[o]
        );
    }
}

#[test]
fn conv2d_pointwise_simd_consistency() {
    // 2 input channels, 11 spatial (SIMD boundary)
    let input = NdTensor::from_vec((0..22).map(|x| x as f32 * 0.1).collect(), &[1, 2, 1, 11]);
    let weight = NdTensor::from_vec(vec![0.5, -0.3], &[1, 2, 1, 1]);
    let bias = NdTensor::from_vec(vec![0.7], &[1]);
    let out = conv2d_pointwise(&input, &weight, Some(&bias));

    for i in 0..11 {
        let expected = input.data[i] * 0.5 + input.data[11 + i] * (-0.3) + 0.7;
        assert!(
            (out.data[i] - expected).abs() < 1e-5,
            "pointwise mismatch at {i}: got {}, expected {expected}",
            out.data[i]
        );
    }
}
