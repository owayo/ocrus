use wide::f32x8;

use crate::tensor::NdTensor;

/// BatchNorm parameters for a single channel.
pub struct BnParams {
    pub gamma: f32,
    pub beta: f32,
    pub running_mean: f32,
    pub running_var: f32,
    pub eps: f32,
}

/// Apply BatchNorm to a 4D tensor (N, C, H, W) in-place.
pub fn batchnorm_inplace(tensor: &mut NdTensor<f32>, params: &[BnParams]) {
    assert_eq!(tensor.ndim(), 4);
    let n = tensor.shape[0];
    let c = tensor.shape[1];
    let h = tensor.shape[2];
    let w = tensor.shape[3];
    assert_eq!(params.len(), c);

    let spatial = h * w;

    for batch in 0..n {
        for (ch, p) in params.iter().enumerate().take(c) {
            let scale = p.gamma / (p.running_var + p.eps).sqrt();
            let bias = p.beta - p.running_mean * scale;

            let offset = (batch * c + ch) * spatial;
            let slice = &mut tensor.data[offset..offset + spatial];

            let scale_v = f32x8::splat(scale);
            let bias_v = f32x8::splat(bias);
            let chunks = spatial / 8;

            for i in 0..chunks {
                let o = i * 8;
                let v = f32x8::from(&slice[o..o + 8]);
                let result = v * scale_v + bias_v;
                let arr: [f32; 8] = result.into();
                slice[o..o + 8].copy_from_slice(&arr);
            }
            for val in slice.iter_mut().skip(chunks * 8) {
                *val = *val * scale + bias;
            }
        }
    }
}

/// Fuse BatchNorm into Conv weights offline.
/// Modifies conv_weight and conv_bias in-place.
pub fn fuse_bn_into_conv(
    conv_weight: &mut [f32],
    conv_bias: &mut [f32],
    params: &[BnParams],
    out_channels: usize,
    kernel_size: usize,
) {
    assert_eq!(params.len(), out_channels);
    assert_eq!(conv_bias.len(), out_channels);

    for ch in 0..out_channels {
        let p = &params[ch];
        let scale = p.gamma / (p.running_var + p.eps).sqrt();

        let weight_offset = ch * kernel_size;
        for i in 0..kernel_size {
            conv_weight[weight_offset + i] *= scale;
        }
        conv_bias[ch] = (conv_bias[ch] - p.running_mean) * scale + p.beta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batchnorm_identity() {
        // gamma=1, beta=0, mean=0, var=1, eps=0 → identity transform
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let expected = data.clone();
        let mut t = NdTensor::from_vec(data, &[1, 1, 4, 4]);
        let params = vec![BnParams {
            gamma: 1.0,
            beta: 0.0,
            running_mean: 0.0,
            running_var: 1.0,
            eps: 0.0,
        }];
        batchnorm_inplace(&mut t, &params);
        for (a, b) in t.data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_batchnorm_scale_shift() {
        // gamma=2, beta=1, mean=0, var=1, eps=0 → y = 2x + 1
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let mut t = NdTensor::from_vec(data, &[1, 1, 2, 2]);
        let params = vec![BnParams {
            gamma: 2.0,
            beta: 1.0,
            running_mean: 0.0,
            running_var: 1.0,
            eps: 0.0,
        }];
        batchnorm_inplace(&mut t, &params);
        assert!((t.data[0] - 1.0).abs() < 1e-6);
        assert!((t.data[1] - 3.0).abs() < 1e-6);
        assert!((t.data[2] - 5.0).abs() < 1e-6);
        assert!((t.data[3] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_fuse_bn() {
        let mut weights = vec![1.0, 2.0, 3.0, 4.0]; // 2 output channels, kernel_size=2
        let mut bias = vec![0.0, 0.0];
        let params = vec![
            BnParams {
                gamma: 2.0,
                beta: 1.0,
                running_mean: 0.5,
                running_var: 1.0,
                eps: 0.0,
            },
            BnParams {
                gamma: 1.0,
                beta: 0.0,
                running_mean: 0.0,
                running_var: 4.0,
                eps: 0.0,
            },
        ];
        fuse_bn_into_conv(&mut weights, &mut bias, &params, 2, 2);
        // Channel 0: scale = 2/1 = 2, weights *= 2, bias = (0 - 0.5)*2 + 1 = 0
        assert!((weights[0] - 2.0).abs() < 1e-6);
        assert!((weights[1] - 4.0).abs() < 1e-6);
        assert!((bias[0] - 0.0).abs() < 1e-6);
        // Channel 1: scale = 1/2 = 0.5, weights *= 0.5, bias = (0 - 0)*0.5 + 0 = 0
        assert!((weights[2] - 1.5).abs() < 1e-6);
        assert!((weights[3] - 2.0).abs() < 1e-6);
        assert!((bias[1] - 0.0).abs() < 1e-6);
    }
}
