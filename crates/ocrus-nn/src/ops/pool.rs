use crate::tensor::NdTensor;

/// MaxPool2d on a 4D tensor (N, C, H, W).
pub fn max_pool2d(
    input: &NdTensor<f32>,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> NdTensor<f32> {
    assert_eq!(input.ndim(), 4);
    let (n, c, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let out_h = (h + 2 * pad_h - kernel_h) / stride_h + 1;
    let out_w = (w + 2 * pad_w - kernel_w) / stride_w + 1;

    let mut output = NdTensor::zeros(&[n, c, out_h, out_w]);

    for batch in 0..n {
        for ch in 0..c {
            let in_offset = (batch * c + ch) * h * w;
            let out_offset = (batch * c + ch) * out_h * out_w;

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let ih = (oh * stride_h + kh).wrapping_sub(pad_h);
                            let iw = (ow * stride_w + kw).wrapping_sub(pad_w);
                            if ih < h && iw < w {
                                let val = input.data[in_offset + ih * w + iw];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    output.data[out_offset + oh * out_w + ow] = max_val;
                }
            }
        }
    }
    output
}

/// AvgPool2d on a 4D tensor (N, C, H, W).
pub fn avg_pool2d(
    input: &NdTensor<f32>,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> NdTensor<f32> {
    assert_eq!(input.ndim(), 4);
    let (n, c, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let out_h = (h + 2 * pad_h - kernel_h) / stride_h + 1;
    let out_w = (w + 2 * pad_w - kernel_w) / stride_w + 1;

    let mut output = NdTensor::zeros(&[n, c, out_h, out_w]);

    for batch in 0..n {
        for ch in 0..c {
            let in_offset = (batch * c + ch) * h * w;
            let out_offset = (batch * c + ch) * out_h * out_w;

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    let mut count = 0u32;
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let ih = (oh * stride_h + kh).wrapping_sub(pad_h);
                            let iw = (ow * stride_w + kw).wrapping_sub(pad_w);
                            if ih < h && iw < w {
                                sum += input.data[in_offset + ih * w + iw];
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        output.data[out_offset + oh * out_w + ow] = sum / count as f32;
                    }
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool_2x2() {
        // 1x1x4x4 input, 2x2 kernel, stride 2, no padding
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = NdTensor::from_vec(data, &[1, 1, 4, 4]);
        let output = max_pool2d(&input, 2, 2, 2, 2, 0, 0);
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
        assert_eq!(output.data, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_avg_pool_2x2() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = NdTensor::from_vec(data, &[1, 1, 4, 4]);
        let output = avg_pool2d(&input, 2, 2, 2, 2, 0, 0);
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
        // avg of [1,2,5,6]=3.5, [3,4,7,8]=5.5, [9,10,13,14]=11.5, [11,12,15,16]=13.5
        assert!((output.data[0] - 3.5).abs() < 1e-6);
        assert!((output.data[1] - 5.5).abs() < 1e-6);
        assert!((output.data[2] - 11.5).abs() < 1e-6);
        assert!((output.data[3] - 13.5).abs() < 1e-6);
    }

    #[test]
    fn test_max_pool_with_padding() {
        // 1x1x2x2 input, 2x2 kernel, stride 1, padding 1 → 3x3 output
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = NdTensor::from_vec(data, &[1, 1, 2, 2]);
        let output = max_pool2d(&input, 2, 2, 1, 1, 1, 1);
        assert_eq!(output.shape, vec![1, 1, 3, 3]);
        // Top-left: only input[0,0]=1.0 is valid
        assert!((output.data[0] - 1.0).abs() < 1e-6);
    }
}
