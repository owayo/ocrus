use crate::arena::TensorArena;
use crate::model::{LayerDescriptor, LayerType, OcnnModel};
use crate::ops;
use crate::tensor::NdTensor;
use ocrus_core::OcrusError;
use ocrus_core::error::Result;

/// Compute SAME_UPPER padding for a single spatial dimension.
/// Returns (pad_begin, pad_end).
fn same_upper_pad(in_size: usize, kernel: usize, stride: usize) -> (usize, usize) {
    let out_size = (in_size + stride - 1) / stride;
    let total_pad = ((out_size - 1) * stride + kernel).saturating_sub(in_size);
    let pad_begin = total_pad / 2;
    let pad_end = total_pad - pad_begin;
    (pad_begin, pad_end)
}

/// Pad a 4D tensor (N,C,H,W) with zeros on the spatial dimensions.
/// Asymmetric padding: (pad_top, pad_bottom, pad_left, pad_right).
fn pad_input(
    input: &NdTensor<f32>,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
) -> NdTensor<f32> {
    if pad_top == 0 && pad_bottom == 0 && pad_left == 0 && pad_right == 0 {
        return input.clone();
    }
    let (n, c, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let mut out = NdTensor::zeros(&[n, c, new_h, new_w]);
    for batch in 0..n {
        for ch in 0..c {
            let src_off = (batch * c + ch) * h * w;
            let dst_off = (batch * c + ch) * new_h * new_w;
            for row in 0..h {
                let src_start = src_off + row * w;
                let dst_start = dst_off + (row + pad_top) * new_w + pad_left;
                out.data[dst_start..dst_start + w]
                    .copy_from_slice(&input.data[src_start..src_start + w]);
            }
        }
    }
    out
}

/// Apply SAME_UPPER padding to input, returning (padded_input, 0, 0) so
/// Conv/Pool can run with pad_h=0, pad_w=0.
fn apply_auto_pad(
    input: &NdTensor<f32>,
    auto_pad: u32,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
) -> Option<NdTensor<f32>> {
    if auto_pad == 0 {
        return None;
    }
    let in_h = input.shape[2];
    let in_w = input.shape[3];
    let (ph_begin, ph_end) = same_upper_pad(in_h, kh, sh);
    let (pw_begin, pw_end) = same_upper_pad(in_w, kw, sw);
    if auto_pad == 2 {
        // SAME_LOWER: swap begin/end
        Some(pad_input(input, ph_end, ph_begin, pw_end, pw_begin))
    } else {
        Some(pad_input(input, ph_begin, ph_end, pw_begin, pw_end))
    }
}

/// Execute the model graph, dispatching v1 (sequential) vs v2 (DAG).
pub fn execute(
    model: &OcnnModel,
    input: NdTensor<f32>,
    arena: &mut TensorArena,
) -> Result<NdTensor<f32>> {
    if model.version == 1 {
        return execute_v1(model, input, arena);
    }
    execute_dag(model, input, arena)
}

/// v1 sequential execution (original behavior).
fn execute_v1(
    model: &OcnnModel,
    input: NdTensor<f32>,
    arena: &mut TensorArena,
) -> Result<NdTensor<f32>> {
    let mut current = input;

    for (i, layer) in model.layers.iter().enumerate() {
        current = execute_layer_v1(model, layer, current, arena).map_err(|e| {
            OcrusError::Runtime(format!("Layer {i} ({:?}) failed: {e}", layer.layer_type))
        })?;
    }

    Ok(current)
}

/// v2 DAG execution with register-based tensor routing.
fn execute_dag(
    model: &OcnnModel,
    input: NdTensor<f32>,
    _arena: &mut TensorArena,
) -> Result<NdTensor<f32>> {
    let num_layers = model.layers.len();
    let mut registers: Vec<Option<NdTensor<f32>>> = vec![None; num_layers];

    // Pre-load constants
    let constants: Vec<NdTensor<f32>> = model
        .constants
        .iter()
        .map(|entry| {
            let data = model.constant_f32(entry).to_vec();
            NdTensor::from_vec(data, &entry.shape)
        })
        .collect();

    for (i, layer) in model.layers.iter().enumerate() {
        let result =
            execute_layer_v2(model, layer, i, &input, &registers, &constants).map_err(|e| {
                OcrusError::Runtime(format!("Layer {i} ({:?}) failed: {e}", layer.layer_type))
            })?;
        registers[i] = Some(result);
    }

    // Return last layer's output
    registers
        .into_iter()
        .rev()
        .find_map(|r| r)
        .ok_or_else(|| OcrusError::Runtime("No layers in model".into()))
}

/// Resolve a single input tensor for a v2 layer.
fn resolve_input<'a>(
    layer: &LayerDescriptor,
    input_idx: usize,
    layer_idx: usize,
    graph_input: &'a NdTensor<f32>,
    registers: &'a [Option<NdTensor<f32>>],
    constants: &'a [NdTensor<f32>],
) -> &'a NdTensor<f32> {
    if layer.num_inputs == 0 {
        // v1-compatible: use previous layer's output (or graph input for layer 0)
        if layer_idx == 0 {
            graph_input
        } else {
            registers[layer_idx - 1].as_ref().expect("missing register")
        }
    } else {
        let ref_idx = layer.inputs[input_idx];
        if ref_idx < 0 {
            // Constant reference: -(idx+1)
            let const_idx = (-(ref_idx + 1)) as usize;
            &constants[const_idx]
        } else {
            let li = ref_idx as usize;
            if li == layer_idx {
                // Self-reference means graph input (for first layer typically)
                graph_input
            } else {
                registers[li]
                    .as_ref()
                    .unwrap_or_else(|| panic!("Layer {layer_idx}: input ref {li} not yet computed"))
            }
        }
    }
}

/// Collect multiple input tensors for a v2 layer.
fn resolve_inputs(
    layer: &LayerDescriptor,
    layer_idx: usize,
    graph_input: &NdTensor<f32>,
    registers: &[Option<NdTensor<f32>>],
    constants: &[NdTensor<f32>],
) -> Vec<NdTensor<f32>> {
    let count = layer.num_inputs as usize;
    (0..count)
        .map(|i| resolve_input(layer, i, layer_idx, graph_input, registers, constants).clone())
        .collect()
}

#[allow(clippy::too_many_lines)]
fn execute_layer_v2(
    model: &OcnnModel,
    layer: &LayerDescriptor,
    layer_idx: usize,
    graph_input: &NdTensor<f32>,
    registers: &[Option<NdTensor<f32>>],
    constants: &[NdTensor<f32>],
) -> Result<NdTensor<f32>> {
    match layer.layer_type {
        // === v1 ops (use first input) ===
        LayerType::ReLU => {
            let mut t =
                resolve_input(layer, 0, layer_idx, graph_input, registers, constants).clone();
            ops::relu::relu_inplace(&mut t);
            Ok(t)
        }
        LayerType::HardSwish => {
            let mut t =
                resolve_input(layer, 0, layer_idx, graph_input, registers, constants).clone();
            ops::relu::hard_swish_inplace(&mut t);
            Ok(t)
        }
        LayerType::Conv2d => {
            let input_raw = resolve_input(layer, 0, layer_idx, graph_input, registers, constants);
            let weights = model.layer_weights_f32(layer);
            let c = &layer.config;
            let (cout, cin, kh, kw) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize);
            let (sh, sw, ph, pw) = (c[4] as usize, c[5] as usize, c[6] as usize, c[7] as usize);
            let has_bias = c[8] != 0;
            let auto_pad = c[9];

            // Apply SAME_UPPER/SAME_LOWER padding if needed
            let padded = apply_auto_pad(input_raw, auto_pad, kh, kw, sh, sw);
            let input = padded.as_ref().unwrap_or(input_raw);
            let (ph, pw) = if auto_pad != 0 { (0, 0) } else { (ph, pw) };

            let w_size = cout * cin * kh * kw;
            let w_tensor = NdTensor::from_vec(weights[..w_size].to_vec(), &[cout, cin, kh, kw]);
            let b_tensor = if has_bias {
                Some(NdTensor::from_vec(
                    weights[w_size..w_size + cout].to_vec(),
                    &[cout],
                ))
            } else {
                None
            };

            if kh == 1 && kw == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0 {
                Ok(ops::conv2d::conv2d_pointwise(
                    input,
                    &w_tensor,
                    b_tensor.as_ref(),
                ))
            } else {
                Ok(ops::conv2d::conv2d_general(
                    input,
                    &w_tensor,
                    b_tensor.as_ref(),
                    sh,
                    sw,
                    ph,
                    pw,
                ))
            }
        }
        LayerType::ConvDepthwise => {
            let input_raw = resolve_input(layer, 0, layer_idx, graph_input, registers, constants);
            let weights = model.layer_weights_f32(layer);
            let c = &layer.config;
            let (channels, kh, kw) = (c[0] as usize, c[2] as usize, c[3] as usize);
            let (sh, sw, ph, pw) = (c[4] as usize, c[5] as usize, c[6] as usize, c[7] as usize);
            let has_bias = c[8] != 0;
            let auto_pad = c[9];

            let padded = apply_auto_pad(input_raw, auto_pad, kh, kw, sh, sw);
            let input = padded.as_ref().unwrap_or(input_raw);
            let (ph, pw) = if auto_pad != 0 { (0, 0) } else { (ph, pw) };

            let w_size = channels * kh * kw;
            let w_tensor = NdTensor::from_vec(weights[..w_size].to_vec(), &[channels, 1, kh, kw]);
            let b_tensor = if has_bias {
                Some(NdTensor::from_vec(
                    weights[w_size..w_size + channels].to_vec(),
                    &[channels],
                ))
            } else {
                None
            };

            Ok(ops::conv2d::conv2d_depthwise(
                input,
                &w_tensor,
                b_tensor.as_ref(),
                sh,
                sw,
                ph,
                pw,
            ))
        }
        LayerType::BatchNorm => {
            let input = resolve_input(layer, 0, layer_idx, graph_input, registers, constants);
            let weights = model.layer_weights_f32(layer);
            let channels = layer.config[0] as usize;
            let eps = f32::from_bits(layer.config[1]);

            let mut params = Vec::with_capacity(channels);
            for ch in 0..channels {
                params.push(ops::batchnorm::BnParams {
                    gamma: weights[ch],
                    beta: weights[channels + ch],
                    running_mean: weights[2 * channels + ch],
                    running_var: weights[3 * channels + ch],
                    eps,
                });
            }

            let mut t = input.clone();
            ops::batchnorm::batchnorm_inplace(&mut t, &params);
            Ok(t)
        }
        LayerType::MaxPool2d => {
            let input_raw = resolve_input(layer, 0, layer_idx, graph_input, registers, constants);
            let c = &layer.config;
            let (kh, kw, sh, sw) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize);
            let auto_pad = c[6];
            let padded = apply_auto_pad(input_raw, auto_pad, kh, kw, sh, sw);
            let input = padded.as_ref().unwrap_or(input_raw);
            let (ph, pw) = if auto_pad != 0 {
                (0, 0)
            } else {
                (c[4] as usize, c[5] as usize)
            };
            Ok(ops::pool::max_pool2d(input, kh, kw, sh, sw, ph, pw))
        }
        LayerType::AvgPool2d => {
            let input_raw = resolve_input(layer, 0, layer_idx, graph_input, registers, constants);
            let c = &layer.config;
            let (kh, kw, sh, sw) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize);
            let auto_pad = c[6];
            let padded = apply_auto_pad(input_raw, auto_pad, kh, kw, sh, sw);
            let input = padded.as_ref().unwrap_or(input_raw);
            let (ph, pw) = if auto_pad != 0 {
                (0, 0)
            } else {
                (c[4] as usize, c[5] as usize)
            };
            Ok(ops::pool::avg_pool2d(input, kh, kw, sh, sw, ph, pw))
        }
        LayerType::Linear => {
            let input = resolve_input(layer, 0, layer_idx, graph_input, registers, constants);
            let weights = model.layer_weights_f32(layer);
            let c = &layer.config;
            let (out_f, in_f) = (c[0] as usize, c[1] as usize);
            let has_bias = c[2] != 0;

            let w_size = out_f * in_f;
            let w_tensor = NdTensor::from_vec(weights[..w_size].to_vec(), &[out_f, in_f]);
            let b_tensor = if has_bias {
                Some(NdTensor::from_vec(
                    weights[w_size..w_size + out_f].to_vec(),
                    &[out_f],
                ))
            } else {
                None
            };
            Ok(ops::linear::linear(input, &w_tensor, b_tensor.as_ref()))
        }
        LayerType::Reshape => {
            let mut t =
                resolve_input(layer, 0, layer_idx, graph_input, registers, constants).clone();
            let c = &layer.config;
            let ndim = c[0] as usize;
            if ndim == 0 && layer.num_inputs >= 2 {
                // Dynamic shape: read from inputs[1] tensor
                let shape_tensor =
                    resolve_input(layer, 1, layer_idx, graph_input, registers, constants);
                let raw_shape: Vec<u32> = shape_tensor
                    .data
                    .iter()
                    .map(|&v| {
                        let iv = v as i64;
                        if iv < 0 {
                            u32::MAX
                        } else if iv == 0 {
                            0
                        } else {
                            iv as u32
                        }
                    })
                    .collect();
                ops::reshape::reshape_dynamic(&mut t, &raw_shape);
            } else {
                let raw_shape: Vec<u32> = (0..ndim).map(|i| c[1 + i]).collect();
                ops::reshape::reshape_dynamic(&mut t, &raw_shape);
            }
            Ok(t)
        }
        LayerType::Flatten => {
            let mut t =
                resolve_input(layer, 0, layer_idx, graph_input, registers, constants).clone();
            let c = &layer.config;
            ops::reshape::flatten(&mut t, c[0] as usize, c[1] as usize);
            Ok(t)
        }
        LayerType::Transpose => {
            let input = resolve_input(layer, 0, layer_idx, graph_input, registers, constants);
            let c = &layer.config;
            let ndim = c[0] as usize;
            if ndim > 2 {
                // General permutation: config = [ndim, perm[0], perm[1], ...]
                let perm: Vec<usize> = (0..ndim).map(|i| c[1 + i] as usize).collect();
                Ok(input.transpose_perm(&perm))
            } else {
                // v1 compat: config = [dim0, dim1]
                Ok(ops::reshape::transpose(input, c[0] as usize, c[1] as usize))
            }
        }

        // === v2 binary ops ===
        LayerType::Add => {
            let inputs = resolve_inputs(layer, layer_idx, graph_input, registers, constants);
            Ok(ops::binary::add(&inputs[0], &inputs[1]))
        }
        LayerType::Sub => {
            let inputs = resolve_inputs(layer, layer_idx, graph_input, registers, constants);
            Ok(ops::binary::sub(&inputs[0], &inputs[1]))
        }
        LayerType::Mul => {
            let inputs = resolve_inputs(layer, layer_idx, graph_input, registers, constants);
            Ok(ops::binary::mul(&inputs[0], &inputs[1]))
        }
        LayerType::Div => {
            let inputs = resolve_inputs(layer, layer_idx, graph_input, registers, constants);
            Ok(ops::binary::div(&inputs[0], &inputs[1]))
        }
        LayerType::MatMul => {
            let inputs = resolve_inputs(layer, layer_idx, graph_input, registers, constants);
            Ok(ops::matmul::matmul(&inputs[0], &inputs[1]))
        }
        LayerType::Pow => {
            let inputs = resolve_inputs(layer, layer_idx, graph_input, registers, constants);
            Ok(ops::math::pow_tensor(&inputs[0], &inputs[1]))
        }

        // === v2 unary ops ===
        LayerType::Sigmoid => {
            let mut t =
                resolve_input(layer, 0, layer_idx, graph_input, registers, constants).clone();
            ops::activation::sigmoid_inplace(&mut t);
            Ok(t)
        }
        LayerType::Softmax => {
            let input = resolve_input(layer, 0, layer_idx, graph_input, registers, constants);
            let raw_axis = layer.config[0] as i32;
            let axis = if raw_axis < 0 {
                (input.ndim() as i32 + raw_axis) as usize
            } else {
                raw_axis as usize
            };
            Ok(ops::activation::softmax(input, axis))
        }
        LayerType::Sqrt => {
            let mut t =
                resolve_input(layer, 0, layer_idx, graph_input, registers, constants).clone();
            ops::math::sqrt_inplace(&mut t);
            Ok(t)
        }

        // === v2 tensor manipulation ops ===
        LayerType::Concat => {
            let num = layer.num_inputs as usize;
            let raw_axis = layer.config[0] as i32;
            if num <= 4 {
                let inputs = resolve_inputs(layer, layer_idx, graph_input, registers, constants);
                let ndim = inputs[0].ndim();
                let axis = if raw_axis < 0 {
                    (ndim as i32 + raw_axis) as usize
                } else {
                    raw_axis as usize
                };
                let refs: Vec<&NdTensor<f32>> = inputs.iter().collect();
                Ok(ops::tensor_ops::concat(&refs, axis))
            } else {
                // >4 inputs: first 4 in descriptor, rest in weight data as i32 array
                let mut all_inputs = Vec::with_capacity(num);
                for i in 0..4 {
                    all_inputs.push(
                        resolve_input(layer, i, layer_idx, graph_input, registers, constants)
                            .clone(),
                    );
                }
                let extra_refs = model.layer_weights_i32(layer);
                for &ref_idx in extra_refs {
                    let t = if ref_idx < 0 {
                        let ci = (-(ref_idx + 1)) as usize;
                        constants[ci].clone()
                    } else {
                        let li = ref_idx as usize;
                        registers[li]
                            .as_ref()
                            .unwrap_or_else(|| {
                                panic!("Layer {layer_idx}: concat ref {li} not computed")
                            })
                            .clone()
                    };
                    all_inputs.push(t);
                }
                let ndim = all_inputs[0].ndim();
                let axis = if raw_axis < 0 {
                    (ndim as i32 + raw_axis) as usize
                } else {
                    raw_axis as usize
                };
                let refs: Vec<&NdTensor<f32>> = all_inputs.iter().collect();
                Ok(ops::tensor_ops::concat(&refs, axis))
            }
        }
        LayerType::Slice => {
            let input = resolve_input(layer, 0, layer_idx, graph_input, registers, constants);
            let c = &layer.config;
            let raw_axis = c[0] as i32;
            let axis = if raw_axis < 0 {
                (input.ndim() as i32 + raw_axis) as usize
            } else {
                raw_axis as usize
            };
            let start = c[1] as i32 as i64;
            let end = c[2] as i32 as i64;
            let step = if c[3] == 0 { 1i64 } else { c[3] as i32 as i64 };
            Ok(ops::tensor_ops::slice_tensor(input, axis, start, end, step))
        }
        LayerType::Squeeze => {
            let mut t =
                resolve_input(layer, 0, layer_idx, graph_input, registers, constants).clone();
            let c = &layer.config;
            let num_axes = c[0] as usize;
            let ndim = t.ndim();
            let axes: Vec<usize> = (0..num_axes)
                .map(|i| {
                    let a = c[1 + i] as i32;
                    if a < 0 {
                        (ndim as i32 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            ops::tensor_ops::squeeze(&mut t, &axes);
            Ok(t)
        }
        LayerType::Unsqueeze => {
            let mut t =
                resolve_input(layer, 0, layer_idx, graph_input, registers, constants).clone();
            let c = &layer.config;
            let num_axes = c[0] as usize;
            let out_ndim = t.ndim() + num_axes;
            let axes: Vec<usize> = (0..num_axes)
                .map(|i| {
                    let a = c[1 + i] as i32;
                    if a < 0 {
                        (out_ndim as i32 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            ops::tensor_ops::unsqueeze(&mut t, &axes);
            Ok(t)
        }

        // === v2 reduce ops ===
        LayerType::ReduceMean => {
            let input = resolve_input(layer, 0, layer_idx, graph_input, registers, constants);
            let c = &layer.config;
            let num_axes = c[0] as usize;
            let ndim = input.ndim();
            let axes: Vec<usize> = (0..num_axes)
                .map(|i| {
                    let a = c[1 + i] as i32;
                    if a < 0 {
                        (ndim as i32 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            let keepdims = c[9] != 0;
            Ok(ops::reduce::reduce_mean(input, &axes, keepdims))
        }

        // === v2 normalization ops ===
        LayerType::LayerNorm => {
            let inputs = resolve_inputs(layer, layer_idx, graph_input, registers, constants);
            let c = &layer.config;
            let axis = c[0] as usize;
            let eps = f32::from_bits(c[1]);
            Ok(ops::layernorm::layer_norm(
                &inputs[0], &inputs[1], &inputs[2], axis, eps,
            ))
        }

        // === v2 shape ===
        LayerType::Shape => {
            let input = resolve_input(layer, 0, layer_idx, graph_input, registers, constants);
            let shape_data: Vec<f32> = input.shape.iter().map(|&d| d as f32).collect();
            let len = shape_data.len();
            Ok(NdTensor::from_vec(shape_data, &[len]))
        }

        // === v2 gather ===
        LayerType::Gather => {
            let inputs = resolve_inputs(layer, layer_idx, graph_input, registers, constants);
            let raw_axis = layer.config[0] as i32;
            let axis = if raw_axis < 0 {
                (inputs[0].ndim() as i32 + raw_axis) as usize
            } else {
                raw_axis as usize
            };
            Ok(ops::gather::gather(&inputs[0], &inputs[1], axis))
        }
    }
}

#[allow(clippy::too_many_lines)]
fn execute_layer_v1(
    model: &OcnnModel,
    layer: &LayerDescriptor,
    input: NdTensor<f32>,
    _arena: &mut TensorArena,
) -> Result<NdTensor<f32>> {
    match layer.layer_type {
        LayerType::ReLU => {
            let mut t = input;
            ops::relu::relu_inplace(&mut t);
            Ok(t)
        }
        LayerType::HardSwish => {
            let mut t = input;
            ops::relu::hard_swish_inplace(&mut t);
            Ok(t)
        }
        LayerType::Conv2d => {
            let weights = model.layer_weights_f32(layer);
            let c = &layer.config;
            let (cout, cin, kh, kw) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize);
            let (sh, sw, ph, pw) = (c[4] as usize, c[5] as usize, c[6] as usize, c[7] as usize);
            let has_bias = c[8] != 0;

            let w_size = cout * cin * kh * kw;
            let w_tensor = NdTensor::from_vec(weights[..w_size].to_vec(), &[cout, cin, kh, kw]);
            let b_tensor = if has_bias {
                Some(NdTensor::from_vec(
                    weights[w_size..w_size + cout].to_vec(),
                    &[cout],
                ))
            } else {
                None
            };

            if kh == 1 && kw == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0 {
                Ok(ops::conv2d::conv2d_pointwise(
                    &input,
                    &w_tensor,
                    b_tensor.as_ref(),
                ))
            } else {
                Ok(ops::conv2d::conv2d_general(
                    &input,
                    &w_tensor,
                    b_tensor.as_ref(),
                    sh,
                    sw,
                    ph,
                    pw,
                ))
            }
        }
        LayerType::ConvDepthwise => {
            let weights = model.layer_weights_f32(layer);
            let c = &layer.config;
            let (channels, kh, kw) = (c[0] as usize, c[2] as usize, c[3] as usize);
            let (sh, sw, ph, pw) = (c[4] as usize, c[5] as usize, c[6] as usize, c[7] as usize);
            let has_bias = c[8] != 0;

            let w_size = channels * kh * kw;
            let w_tensor = NdTensor::from_vec(weights[..w_size].to_vec(), &[channels, 1, kh, kw]);
            let b_tensor = if has_bias {
                Some(NdTensor::from_vec(
                    weights[w_size..w_size + channels].to_vec(),
                    &[channels],
                ))
            } else {
                None
            };

            Ok(ops::conv2d::conv2d_depthwise(
                &input,
                &w_tensor,
                b_tensor.as_ref(),
                sh,
                sw,
                ph,
                pw,
            ))
        }
        LayerType::BatchNorm => {
            let weights = model.layer_weights_f32(layer);
            let channels = layer.config[0] as usize;
            let eps = f32::from_bits(layer.config[1]);

            let mut params = Vec::with_capacity(channels);
            for ch in 0..channels {
                params.push(ops::batchnorm::BnParams {
                    gamma: weights[ch],
                    beta: weights[channels + ch],
                    running_mean: weights[2 * channels + ch],
                    running_var: weights[3 * channels + ch],
                    eps,
                });
            }

            let mut t = input;
            ops::batchnorm::batchnorm_inplace(&mut t, &params);
            Ok(t)
        }
        LayerType::MaxPool2d => {
            let c = &layer.config;
            Ok(ops::pool::max_pool2d(
                &input,
                c[0] as usize,
                c[1] as usize,
                c[2] as usize,
                c[3] as usize,
                c[4] as usize,
                c[5] as usize,
            ))
        }
        LayerType::AvgPool2d => {
            let c = &layer.config;
            Ok(ops::pool::avg_pool2d(
                &input,
                c[0] as usize,
                c[1] as usize,
                c[2] as usize,
                c[3] as usize,
                c[4] as usize,
                c[5] as usize,
            ))
        }
        LayerType::Linear => {
            let weights = model.layer_weights_f32(layer);
            let c = &layer.config;
            let (out_f, in_f) = (c[0] as usize, c[1] as usize);
            let has_bias = c[2] != 0;

            let w_size = out_f * in_f;
            let w_tensor = NdTensor::from_vec(weights[..w_size].to_vec(), &[out_f, in_f]);
            let b_tensor = if has_bias {
                Some(NdTensor::from_vec(
                    weights[w_size..w_size + out_f].to_vec(),
                    &[out_f],
                ))
            } else {
                None
            };
            Ok(ops::linear::linear(&input, &w_tensor, b_tensor.as_ref()))
        }
        LayerType::Reshape => {
            let c = &layer.config;
            let ndim = c[0] as usize;
            let new_shape: Vec<usize> = (0..ndim).map(|i| c[1 + i] as usize).collect();
            let mut t = input;
            ops::reshape::reshape(&mut t, &new_shape);
            Ok(t)
        }
        LayerType::Flatten => {
            let c = &layer.config;
            let mut t = input;
            ops::reshape::flatten(&mut t, c[0] as usize, c[1] as usize);
            Ok(t)
        }
        LayerType::Transpose => {
            let c = &layer.config;
            Ok(ops::reshape::transpose(
                &input,
                c[0] as usize,
                c[1] as usize,
            ))
        }
        // v2 ops should not appear in v1 models
        _ => Err(OcrusError::Runtime(format!(
            "v2 op {:?} in v1 model",
            layer.layer_type
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ConstantDef, LayerDescriptor, LayerType, build_ocnn, build_ocnn_v2};
    use memmap2::MmapMut;

    fn make_mmap(data: &[u8]) -> memmap2::Mmap {
        let mut mm = MmapMut::map_anon(data.len()).unwrap();
        mm.copy_from_slice(data);
        mm.make_read_only().unwrap()
    }

    #[test]
    fn test_relu_then_linear() {
        let relu_desc = LayerDescriptor {
            layer_type: LayerType::ReLU,
            num_inputs: 0,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
            inputs: [0; 4],
        };

        let linear_weights: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 10.0, 20.0];
        let weight_bytes: Vec<u8> = linear_weights
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let linear_desc = LayerDescriptor {
            layer_type: LayerType::Linear,
            num_inputs: 0,
            param_offset: 0,
            param_size: weight_bytes.len() as u64,
            config: [2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
            inputs: [0; 4],
        };

        let data = build_ocnn(&[(relu_desc, &[]), (linear_desc, &weight_bytes)]);
        let model = OcnnModel::from_mmap(make_mmap(&data)).unwrap();

        let input = NdTensor::from_vec(vec![-1.0, 3.0], &[1, 2]);
        let mut arena = TensorArena::new();
        let output = execute(&model, input, &mut arena).unwrap();

        assert_eq!(output.shape, vec![1, 2]);
        assert!((output.data[0] - 10.0).abs() < 1e-6);
        assert!((output.data[1] - 23.0).abs() < 1e-6);
    }

    #[test]
    fn test_hardswish_graph() {
        let hs_desc = LayerDescriptor {
            layer_type: LayerType::HardSwish,
            num_inputs: 0,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
            inputs: [0; 4],
        };
        let data = build_ocnn(&[(hs_desc, &[])]);
        let model = OcnnModel::from_mmap(make_mmap(&data)).unwrap();

        let input = NdTensor::from_vec(vec![0.0, 3.0, 6.0, -4.0], &[1, 4]);
        let mut arena = TensorArena::new();
        let output = execute(&model, input, &mut arena).unwrap();
        assert_eq!(output.shape, vec![1, 4]);
        assert!((output.data[0] - 0.0).abs() < 1e-6);
        assert!((output.data[1] - 3.0).abs() < 1e-6);
        assert!((output.data[2] - 6.0).abs() < 1e-6);
        assert!((output.data[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dag_add_with_constant() {
        // v2 model: Add(input, constant)
        // constant = [10.0, 20.0]
        let const_data: Vec<u8> = [10.0f32, 20.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let add_desc = LayerDescriptor {
            layer_type: LayerType::Add,
            num_inputs: 2,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
            inputs: [0, -1, 0, 0], // input 0 = layer 0 (self = graph input), input 1 = constant 0
        };

        let data = build_ocnn_v2(
            &[ConstantDef {
                shape: &[2],
                data: &const_data,
            }],
            &[(add_desc, &[])],
        );
        let model = OcnnModel::from_mmap(make_mmap(&data)).unwrap();
        assert_eq!(model.version, 2);

        let input = NdTensor::from_vec(vec![1.0, 2.0], &[2]);
        let mut arena = TensorArena::new();
        let output = execute(&model, input, &mut arena).unwrap();
        assert_eq!(output.data, vec![11.0, 22.0]);
    }

    #[test]
    fn test_dag_reshape_dynamic() {
        // v2 model: Reshape with -1
        let reshape_desc = LayerDescriptor {
            layer_type: LayerType::Reshape,
            num_inputs: 1,
            param_offset: 0,
            param_size: 0,
            // ndim=2, shape=[0, 0xFFFFFFFF] => [input_dim0, inferred]
            config: [2, 0, u32::MAX, 0, 0, 0, 0, 0, 0, 0],
            inputs: [0, 0, 0, 0], // self-reference = graph input
        };

        let data = build_ocnn_v2(&[], &[(reshape_desc, &[])]);
        let model = OcnnModel::from_mmap(make_mmap(&data)).unwrap();

        let input = NdTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let mut arena = TensorArena::new();
        let output = execute(&model, input, &mut arena).unwrap();
        assert_eq!(output.shape, vec![2, 3]); // 0 copies dim, -1 infers
    }

    #[test]
    fn test_dag_sigmoid_then_matmul() {
        // v2 model: sigmoid(input) -> matmul(result, constant_matrix)
        let const_data: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let sigmoid_desc = LayerDescriptor {
            layer_type: LayerType::Sigmoid,
            num_inputs: 1,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
            inputs: [0, 0, 0, 0], // self-ref = graph input (layer 0)
        };

        let matmul_desc = LayerDescriptor {
            layer_type: LayerType::MatMul,
            num_inputs: 2,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
            inputs: [0, -1, 0, 0], // input 0 = layer 0 (sigmoid), input 1 = constant 0
        };

        let data = build_ocnn_v2(
            &[ConstantDef {
                shape: &[2, 2],
                data: &const_data,
            }],
            &[(sigmoid_desc, &[]), (matmul_desc, &[])],
        );
        let model = OcnnModel::from_mmap(make_mmap(&data)).unwrap();

        let input = NdTensor::from_vec(vec![0.0, 0.0], &[1, 2]);
        let mut arena = TensorArena::new();
        let output = execute(&model, input, &mut arena).unwrap();
        // sigmoid(0) = 0.5, matmul with identity = [0.5, 0.5]
        assert_eq!(output.shape, vec![1, 2]);
        assert!((output.data[0] - 0.5).abs() < 1e-6);
        assert!((output.data[1] - 0.5).abs() < 1e-6);
    }
}
