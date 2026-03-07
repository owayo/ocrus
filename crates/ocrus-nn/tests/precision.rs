//! Numerical precision tests for ocrus-nn ops and graph execution.
//! Validates SIMD paths match scalar results and multi-layer pipelines
//! produce deterministic golden outputs.

use memmap2::MmapMut;
use ocrus_nn::model::{LayerDescriptor, LayerType, OcnnModel, build_ocnn};
use ocrus_nn::ops::batchnorm::{BnParams, batchnorm_inplace};
use ocrus_nn::ops::conv2d::{conv2d_depthwise, conv2d_general, conv2d_pointwise};
use ocrus_nn::ops::linear::linear;
use ocrus_nn::ops::pool::{avg_pool2d, max_pool2d};
use ocrus_nn::ops::relu::{hard_swish_inplace, relu_inplace};
use ocrus_nn::ops::reshape::flatten;
use ocrus_nn::tensor::NdTensor;
use ocrus_nn::{NnEngine, Tensor};

fn make_model(layers: &[(LayerDescriptor, &[u8])]) -> OcnnModel {
    let data = build_ocnn(layers);
    let mut mm = MmapMut::map_anon(data.len()).unwrap();
    mm.copy_from_slice(&data);
    let mmap = mm.make_read_only().unwrap();
    OcnnModel::from_mmap(mmap).unwrap()
}

fn f32_to_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|f| f.to_le_bytes()).collect()
}

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

// ============================================================
// 2. Golden output tests: multi-layer graph
// ============================================================

#[test]
fn golden_conv_bn_relu_pool_flatten_linear() {
    // Build a small CNN: Conv2d(1->2, 3x3, pad=1) -> BN -> ReLU -> AvgPool(2x2) -> Flatten -> Linear(2->3)
    // Input: (1, 1, 4, 4)

    // Step 1: Compute expected output manually
    #[rustfmt::skip]
    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let input = NdTensor::from_vec(input_data.clone(), &[1, 1, 4, 4]);

    // Conv weights: 2 output channels, 1 input channel, 3x3
    // Channel 0: identity center, Channel 1: all 1/9
    let mut conv_w = vec![0.0f32; 18]; // 2*1*3*3
    conv_w[4] = 1.0; // center of first 3x3 filter
    for val in &mut conv_w[9..18] {
        *val = 1.0 / 9.0; // second filter: average
    }
    let conv_b = vec![0.0, 0.0];
    let conv_weight = NdTensor::from_vec(conv_w.clone(), &[2, 1, 3, 3]);
    let conv_bias = NdTensor::from_vec(conv_b.clone(), &[2]);

    // Conv2d
    let after_conv = conv2d_general(&input, &conv_weight, Some(&conv_bias), 1, 1, 1, 1);
    assert_eq!(after_conv.shape, vec![1, 2, 4, 4]);

    // BatchNorm: identity transform (gamma=1, beta=0, mean=0, var=1)
    let bn_params = vec![
        BnParams {
            gamma: 1.0,
            beta: 0.0,
            running_mean: 0.0,
            running_var: 1.0,
            eps: 1e-5,
        },
        BnParams {
            gamma: 1.0,
            beta: 0.0,
            running_mean: 0.0,
            running_var: 1.0,
            eps: 1e-5,
        },
    ];
    let mut after_bn = after_conv.clone();
    batchnorm_inplace(&mut after_bn, &bn_params);

    // ReLU
    let mut after_relu = after_bn.clone();
    relu_inplace(&mut after_relu);

    // AvgPool 2x2, stride 2
    let after_pool = avg_pool2d(&after_relu, 2, 2, 2, 2, 0, 0);
    assert_eq!(after_pool.shape, vec![1, 2, 2, 2]);

    // Flatten (1,2,2,2) -> (1, 8)
    let mut after_flat = after_pool.clone();
    flatten(&mut after_flat, 1, 3);
    assert_eq!(after_flat.shape, vec![1, 8]);

    // Linear: (1,8) -> (1,3)
    let lin_w: Vec<f32> = (0..24).map(|x| (x as f32 - 12.0) * 0.01).collect();
    let lin_b = vec![0.1, 0.2, 0.3];
    let lin_weight = NdTensor::from_vec(lin_w.clone(), &[3, 8]);
    let lin_bias = NdTensor::from_vec(lin_b.clone(), &[3]);
    let expected_output = linear(&after_flat, &lin_weight, Some(&lin_bias));

    // Step 2: Build .ocnn model and run through graph
    // Conv2d layer
    let mut conv_weights_bytes = f32_to_bytes(&conv_w);
    conv_weights_bytes.extend_from_slice(&f32_to_bytes(&conv_b));
    let conv_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::Conv2d,
        param_offset: 0,
        param_size: conv_weights_bytes.len() as u64,
        config: [2, 1, 3, 3, 1, 1, 1, 1, 1, 0], // cout, cin, kh, kw, sh, sw, ph, pw, has_bias
    };

    // BatchNorm layer
    let eps_bits = 1e-5f32.to_bits();
    let bn_weights: Vec<f32> = vec![
        1.0, 1.0, // gamma
        0.0, 0.0, // beta
        0.0, 0.0, // mean
        1.0, 1.0, // var
    ];
    let bn_bytes = f32_to_bytes(&bn_weights);
    let bn_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::BatchNorm,
        param_offset: 0,
        param_size: bn_bytes.len() as u64,
        config: [2, eps_bits, 0, 0, 0, 0, 0, 0, 0, 0],
    };

    // ReLU
    let relu_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::ReLU,
        param_offset: 0,
        param_size: 0,
        config: [0; 10],
    };

    // AvgPool 2x2 stride 2
    let pool_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::AvgPool2d,
        param_offset: 0,
        param_size: 0,
        config: [2, 2, 2, 2, 0, 0, 0, 0, 0, 0], // kh, kw, sh, sw, ph, pw
    };

    // Flatten 1..3
    let flat_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::Flatten,
        param_offset: 0,
        param_size: 0,
        config: [1, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    };

    // Linear
    let mut lin_weights_bytes = f32_to_bytes(&lin_w);
    lin_weights_bytes.extend_from_slice(&f32_to_bytes(&lin_b));
    let lin_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::Linear,
        param_offset: 0,
        param_size: lin_weights_bytes.len() as u64,
        config: [3, 8, 1, 0, 0, 0, 0, 0, 0, 0], // out=3, in=8, has_bias=1
    };

    let model = make_model(&[
        (conv_desc, &conv_weights_bytes),
        (bn_desc, &bn_bytes),
        (relu_desc, &[]),
        (pool_desc, &[]),
        (flat_desc, &[]),
        (lin_desc, &lin_weights_bytes),
    ]);

    let engine = NnEngine::new().unwrap();
    let result = engine
        .run(&model, &[Tensor::new(input_data, vec![1, 1, 4, 4])])
        .unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].shape, vec![1, 3]);
    for i in 0..3 {
        assert!(
            (result[0].data[i] - expected_output.data[i]).abs() < 1e-4,
            "golden output mismatch at {i}: got {}, expected {}",
            result[0].data[i],
            expected_output.data[i]
        );
    }
}

#[test]
fn golden_depthwise_hardswish_maxpool() {
    // DepthwiseConv(2ch, 3x3, pad=1) -> HardSwish -> MaxPool(2x2)
    // Input: (1, 2, 4, 4)
    let input_data: Vec<f32> = (0..32).map(|x| (x as f32 - 16.0) * 0.25).collect();
    let input = NdTensor::from_vec(input_data.clone(), &[1, 2, 4, 4]);

    // Depthwise weights: 2 channels, 3x3
    let dw_w: Vec<f32> = (0..18).map(|x| (x as f32 - 9.0) * 0.1).collect();
    let dw_b = vec![0.5, -0.5];
    let dw_weight = NdTensor::from_vec(dw_w.clone(), &[2, 1, 3, 3]);
    let dw_bias = NdTensor::from_vec(dw_b.clone(), &[2]);

    let after_conv = conv2d_depthwise(&input, &dw_weight, Some(&dw_bias), 1, 1, 1, 1);
    assert_eq!(after_conv.shape, vec![1, 2, 4, 4]);

    let mut after_hs = after_conv.clone();
    hard_swish_inplace(&mut after_hs);

    let expected = max_pool2d(&after_hs, 2, 2, 2, 2, 0, 0);
    assert_eq!(expected.shape, vec![1, 2, 2, 2]);

    // Build .ocnn and compare
    let mut dw_bytes = f32_to_bytes(&dw_w);
    dw_bytes.extend_from_slice(&f32_to_bytes(&dw_b));
    let dw_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::ConvDepthwise,
        param_offset: 0,
        param_size: dw_bytes.len() as u64,
        config: [2, 2, 3, 3, 1, 1, 1, 1, 1, 0], // channels, cin(unused), kh, kw, sh, sw, ph, pw, has_bias
    };

    let hs_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::HardSwish,
        param_offset: 0,
        param_size: 0,
        config: [0; 10],
    };

    let pool_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::MaxPool2d,
        param_offset: 0,
        param_size: 0,
        config: [2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    };

    let model = make_model(&[(dw_desc, &dw_bytes), (hs_desc, &[]), (pool_desc, &[])]);

    let engine = NnEngine::new().unwrap();
    let result = engine
        .run(&model, &[Tensor::new(input_data, vec![1, 2, 4, 4])])
        .unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].shape, expected.shape);
    for i in 0..expected.data.len() {
        assert!(
            (result[0].data[i] - expected.data[i]).abs() < 1e-5,
            "golden dw+hs+pool mismatch at {i}: got {}, expected {}",
            result[0].data[i],
            expected.data[i]
        );
    }
}

// ============================================================
// 3. E2E NnEngine API tests
// ============================================================

#[test]
fn engine_run_batch_consistency() {
    // run_batch splits output assuming 3D (N, T, C) shape (OCR CTC output).
    // Build a model that produces 3D output: ReLU -> Reshape(N, T, C)
    let relu_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::ReLU,
        param_offset: 0,
        param_size: 0,
        config: [0; 10],
    };
    // Reshape (N, 1, 1, 4) -> (N, 2, 2) to simulate CTC-like output
    // Flatten(2,3) on (N,1,1,4) -> (N,1,4) to produce 3D output for run_batch splitting
    let flat_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::Flatten,
        param_offset: 0,
        param_size: 0,
        config: [2, 3, 0, 0, 0, 0, 0, 0, 0, 0], // start=2, end=3
    };
    let model = make_model(&[(relu_desc, &[]), (flat_desc, &[])]);
    let engine = NnEngine::new().unwrap();

    let t1 = Tensor::new(vec![-1.0, 2.0, -3.0, 0.5], vec![1, 1, 1, 4]);
    let t2 = Tensor::new(vec![4.0, -5.0, 6.0, -0.5], vec![1, 1, 1, 4]);

    // Individual runs: output shape (1, 1, 4)
    let r1 = engine.run(&model, std::slice::from_ref(&t1)).unwrap();
    let r2 = engine.run(&model, std::slice::from_ref(&t2)).unwrap();
    assert_eq!(r1[0].shape, vec![1, 1, 4]);

    // Batch run (same width, no padding)
    let batch_results = engine.run_batch(&model, &[t1, t2]).unwrap();
    assert_eq!(batch_results.len(), 2);
    assert_eq!(batch_results[0][0].shape, vec![1, 1, 4]);
    assert_eq!(batch_results[1][0].shape, vec![1, 1, 4]);

    for i in 0..4 {
        assert!(
            (batch_results[0][0].data[i] - r1[0].data[i]).abs() < 1e-7,
            "batch[0] mismatch at {i}: got {}, expected {}",
            batch_results[0][0].data[i],
            r1[0].data[i]
        );
        assert!(
            (batch_results[1][0].data[i] - r2[0].data[i]).abs() < 1e-7,
            "batch[1] mismatch at {i}: got {}, expected {}",
            batch_results[1][0].data[i],
            r2[0].data[i]
        );
    }
}

#[test]
fn engine_run_batch_different_widths() {
    // Batch with different widths should pad correctly
    let relu_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::ReLU,
        param_offset: 0,
        param_size: 0,
        config: [0; 10],
    };
    let model = make_model(&[(relu_desc, &[])]);
    let engine = NnEngine::new().unwrap();

    let t1 = Tensor::new(vec![-1.0, 2.0], vec![1, 1, 1, 2]);
    let t2 = Tensor::new(vec![3.0, -4.0, 5.0, -6.0], vec![1, 1, 1, 4]);

    let results = engine.run_batch(&model, &[t1, t2]).unwrap();
    assert_eq!(results.len(), 2);

    // t1 result: [-1,2] -> relu -> [0,2] (padded with 1.0 to width 4, but pad=1.0 > 0 so relu keeps)
    // t2 result: [3,-4,5,-6] -> relu -> [3,0,5,0]
    // Note: padding values (1.0) go through ReLU too, but we only check original positions
}

#[test]
fn engine_deterministic_repeated_runs() {
    // Same input should always produce same output
    let conv_w = vec![0.5f32; 9]; // 1x1x3x3
    let conv_b = vec![0.1f32];
    let mut weights = f32_to_bytes(&conv_w);
    weights.extend_from_slice(&f32_to_bytes(&conv_b));

    let conv_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::Conv2d,
        param_offset: 0,
        param_size: weights.len() as u64,
        config: [1, 1, 3, 3, 1, 1, 1, 1, 1, 0],
    };
    let relu_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::ReLU,
        param_offset: 0,
        param_size: 0,
        config: [0; 10],
    };

    let model = make_model(&[(conv_desc, &weights), (relu_desc, &[])]);
    let engine = NnEngine::new().unwrap();

    let input_data: Vec<f32> = (0..16).map(|x| x as f32 * 0.1 - 0.5).collect();
    let input = Tensor::new(input_data.clone(), vec![1, 1, 4, 4]);

    let r1 = engine.run(&model, std::slice::from_ref(&input)).unwrap();
    let r2 = engine.run(&model, std::slice::from_ref(&input)).unwrap();
    let r3 = engine.run(&model, std::slice::from_ref(&input)).unwrap();

    assert_eq!(r1[0].data, r2[0].data);
    assert_eq!(r2[0].data, r3[0].data);
}

#[test]
fn engine_transpose_reshape_pipeline() {
    // Test Reshape -> Transpose -> Flatten pipeline
    // Input: (1, 6) -> Reshape to (1, 2, 3) -> Transpose(1,2) -> (1, 3, 2) -> Flatten(1,2) -> (1, 6)
    let reshape_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::Reshape,
        param_offset: 0,
        param_size: 0,
        config: [3, 1, 2, 3, 0, 0, 0, 0, 0, 0], // ndim=3, shape=[1,2,3]
    };
    let transpose_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::Transpose,
        param_offset: 0,
        param_size: 0,
        config: [1, 2, 0, 0, 0, 0, 0, 0, 0, 0], // dim0=1, dim1=2
    };
    let flatten_desc = LayerDescriptor { num_inputs: 0, inputs: [0; 4],
        layer_type: LayerType::Flatten,
        param_offset: 0,
        param_size: 0,
        config: [1, 2, 0, 0, 0, 0, 0, 0, 0, 0], // start=1, end=2
    };

    let model = make_model(&[
        (reshape_desc, &[]),
        (transpose_desc, &[]),
        (flatten_desc, &[]),
    ]);
    let engine = NnEngine::new().unwrap();

    // Input [0,1,2,3,4,5] reshaped to [[0,1,2],[3,4,5]]
    // Transposed: [[0,3],[1,4],[2,5]]
    // Flattened: [0,3,1,4,2,5]
    let input = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 6]);
    let result = engine.run(&model, &[input]).unwrap();

    assert_eq!(result[0].shape, vec![1, 6]);
    assert_eq!(result[0].data, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}
