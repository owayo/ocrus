use crate::arena::TensorArena;
use crate::model::{LayerDescriptor, LayerType, OcnnModel};
use crate::ops;
use crate::tensor::NdTensor;
use ocrus_core::OcrusError;
use ocrus_core::error::Result;

/// Execute the model graph sequentially
pub fn execute(
    model: &OcnnModel,
    input: NdTensor<f32>,
    arena: &mut TensorArena,
) -> Result<NdTensor<f32>> {
    let mut current = input;

    for (i, layer) in model.layers.iter().enumerate() {
        current = execute_layer(model, layer, current, arena).map_err(|e| {
            OcrusError::Runtime(format!("Layer {i} ({:?}) failed: {e}", layer.layer_type))
        })?;
    }

    Ok(current)
}

#[allow(clippy::too_many_lines)]
fn execute_layer(
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
            // config: [out_ch, in_ch, kh, kw, stride_h, stride_w, pad_h, pad_w, has_bias, 0]
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LayerDescriptor, LayerType, build_ocnn};
    use memmap2::MmapMut;

    fn make_mmap(data: &[u8]) -> memmap2::Mmap {
        let mut mm = MmapMut::map_anon(data.len()).unwrap();
        mm.copy_from_slice(data);
        mm.make_read_only().unwrap()
    }

    #[test]
    fn test_relu_then_linear() {
        // Build a model: ReLU -> Linear(2->2, bias)
        // Weight: identity matrix, bias = [10, 20]
        let relu_desc = LayerDescriptor {
            layer_type: LayerType::ReLU,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
        };

        // Linear weights: [[1,0],[0,1]] + bias [10, 20]
        let linear_weights: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 10.0, 20.0];
        let weight_bytes: Vec<u8> = linear_weights
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let linear_desc = LayerDescriptor {
            layer_type: LayerType::Linear,
            param_offset: 0,
            param_size: weight_bytes.len() as u64,
            config: [2, 2, 1, 0, 0, 0, 0, 0, 0, 0], // out=2, in=2, has_bias=1
        };

        let data = build_ocnn(&[(relu_desc, &[]), (linear_desc, &weight_bytes)]);
        let model = OcnnModel::from_mmap(make_mmap(&data)).unwrap();

        // Input: [[-1, 3]] -> after ReLU: [[0, 3]] -> after Linear: [[10, 23]]
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
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
        };
        let data = build_ocnn(&[(hs_desc, &[])]);
        let model = OcnnModel::from_mmap(make_mmap(&data)).unwrap();

        let input = NdTensor::from_vec(vec![0.0, 3.0, 6.0, -4.0], &[1, 4]);
        let mut arena = TensorArena::new();
        let output = execute(&model, input, &mut arena).unwrap();
        assert_eq!(output.shape, vec![1, 4]);
        // x=0: 0 * (0+3)/6 = 0
        assert!((output.data[0] - 0.0).abs() < 1e-6);
        // x=3: 3 * (3+3)/6 = 3
        assert!((output.data[1] - 3.0).abs() < 1e-6);
        // x=6: 6 (clamped)
        assert!((output.data[2] - 6.0).abs() < 1e-6);
        // x=-4: 0 (clamped)
        assert!((output.data[3] - 0.0).abs() < 1e-6);
    }
}
