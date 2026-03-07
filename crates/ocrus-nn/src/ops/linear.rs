use wide::f32x8;

use crate::tensor::NdTensor;

/// Linear (fully connected) layer: output = input * weight^T + bias
///
/// input: (batch, in_features)
/// weight: (out_features, in_features)
/// bias: (out_features,) or None
pub fn linear(
    input: &NdTensor<f32>,
    weight: &NdTensor<f32>,
    bias: Option<&NdTensor<f32>>,
) -> NdTensor<f32> {
    assert_eq!(input.ndim(), 2);
    assert_eq!(weight.ndim(), 2);
    let batch = input.shape[0];
    let in_features = input.shape[1];
    let out_features = weight.shape[0];
    assert_eq!(weight.shape[1], in_features);

    if let Some(b) = bias {
        assert_eq!(b.shape[0], out_features);
    }

    let mut output = NdTensor::zeros(&[batch, out_features]);

    for b in 0..batch {
        let in_offset = b * in_features;
        let out_offset = b * out_features;
        let in_row = &input.data[in_offset..in_offset + in_features];

        for o in 0..out_features {
            let w_offset = o * in_features;
            let w_row = &weight.data[w_offset..w_offset + in_features];

            // SIMD dot product
            let chunks = in_features / 8;
            let mut acc = f32x8::ZERO;

            for c in 0..chunks {
                let off = c * 8;
                let a = f32x8::from(&in_row[off..off + 8]);
                let b_v = f32x8::from(&w_row[off..off + 8]);
                acc = acc + a * b_v;
            }

            // Reduce SIMD accumulator
            let arr: [f32; 8] = acc.into();
            let mut dot: f32 = arr.iter().sum();

            // Handle remainder
            for i in (chunks * 8)..in_features {
                dot += in_row[i] * w_row[i];
            }

            // Add bias
            if let Some(bias_t) = bias {
                dot += bias_t.data[o];
            }

            output.data[out_offset + o] = dot;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_basic() {
        // input: (1, 3), weight: (2, 3), bias: (2,)
        let input = NdTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        let weight = NdTensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        let bias = NdTensor::from_vec(vec![0.5, -0.5], &[2]);
        let output = linear(&input, &weight, Some(&bias));
        assert_eq!(output.shape, vec![1, 2]);
        assert!((output.data[0] - 1.5).abs() < 1e-6); // 1*1 + 2*0 + 3*0 + 0.5
        assert!((output.data[1] - 1.5).abs() < 1e-6); // 1*0 + 2*1 + 3*0 - 0.5
    }

    #[test]
    fn test_linear_no_bias() {
        let input = NdTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let weight = NdTensor::from_vec(vec![1.0, 1.0], &[1, 2]);
        let output = linear(&input, &weight, None);
        assert_eq!(output.shape, vec![2, 1]);
        assert!((output.data[0] - 3.0).abs() < 1e-6); // 1+2
        assert!((output.data[1] - 7.0).abs() < 1e-6); // 3+4
    }

    #[test]
    fn test_linear_simd_path() {
        // 16 features to exercise SIMD path (2 chunks of 8)
        let in_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let w_data: Vec<f32> = vec![1.0; 16];
        let input = NdTensor::from_vec(in_data, &[1, 16]);
        let weight = NdTensor::from_vec(w_data, &[1, 16]);
        let output = linear(&input, &weight, None);
        // sum of 0..16 = 120
        assert!((output.data[0] - 120.0).abs() < 1e-4);
    }
}
