use wide::f32x8;

use crate::tensor::NdTensor;

/// 1x1 Pointwise convolution: input (N,Cin,H,W), weight (Cout,Cin,1,1), bias (Cout,)
/// Output: (N,Cout,H,W)
pub fn conv2d_pointwise(
    input: &NdTensor<f32>,
    weight: &NdTensor<f32>,
    bias: Option<&NdTensor<f32>>,
) -> NdTensor<f32> {
    assert_eq!(input.ndim(), 4);
    assert_eq!(weight.ndim(), 4);
    let (n, cin, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let cout = weight.shape[0];
    assert_eq!(weight.shape[1], cin);

    let spatial = h * w;
    let mut output = NdTensor::zeros(&[n, cout, h, w]);

    for batch in 0..n {
        for co in 0..cout {
            let b = bias.map_or(0.0, |b| b.data[co]);
            let out_offset = (batch * cout + co) * spatial;
            // Fill with bias
            for i in 0..spatial {
                output.data[out_offset + i] = b;
            }
            // Accumulate: for each input channel, multiply weight and add
            for ci in 0..cin {
                let w_val = weight.data[co * cin + ci]; // weight shape is (Cout,Cin,1,1)
                let in_offset = (batch * cin + ci) * spatial;

                let w_v = f32x8::splat(w_val);
                let chunks = spatial / 8;
                for i in 0..chunks {
                    let o = i * 8;
                    let inv = f32x8::from(&input.data[in_offset + o..in_offset + o + 8]);
                    let outv = f32x8::from(&output.data[out_offset + o..out_offset + o + 8]);
                    let result = outv + inv * w_v;
                    let arr: [f32; 8] = result.into();
                    output.data[out_offset + o..out_offset + o + 8].copy_from_slice(&arr);
                }
                for i in (chunks * 8)..spatial {
                    output.data[out_offset + i] += input.data[in_offset + i] * w_val;
                }
            }
        }
    }
    output
}

/// Depthwise convolution: each channel has its own kernel
/// input (N,C,H,W), weight (C,1,KH,KW), bias (C,)
pub fn conv2d_depthwise(
    input: &NdTensor<f32>,
    weight: &NdTensor<f32>,
    bias: Option<&NdTensor<f32>>,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> NdTensor<f32> {
    assert_eq!(input.ndim(), 4);
    assert_eq!(weight.ndim(), 4);
    let (n, c, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let (kh, kw) = (weight.shape[2], weight.shape[3]);
    assert_eq!(weight.shape[0], c);
    assert_eq!(weight.shape[1], 1);

    let out_h = (h + 2 * pad_h - kh) / stride_h + 1;
    let out_w = (w + 2 * pad_w - kw) / stride_w + 1;
    let mut output = NdTensor::zeros(&[n, c, out_h, out_w]);

    for batch in 0..n {
        for ch in 0..c {
            let b = bias.map_or(0.0, |b| b.data[ch]);
            let in_offset = (batch * c + ch) * h * w;
            let out_offset = (batch * c + ch) * out_h * out_w;
            let w_offset = ch * kh * kw;

            for oh in 0..out_h {
                // SIMD: process 8 output columns at a time
                let chunks = out_w / 8;
                for ow_chunk in 0..chunks {
                    let ow_base = ow_chunk * 8;
                    let mut acc = f32x8::splat(b);

                    for ky in 0..kh {
                        let ih = oh * stride_h + ky;
                        if ih < pad_h || ih - pad_h >= h {
                            continue;
                        }
                        let ih = ih - pad_h;
                        for kx in 0..kw {
                            let w_val = weight.data[w_offset + ky * kw + kx];
                            let w_v = f32x8::splat(w_val);

                            // Gather 8 input values
                            let mut in_vals = [0.0f32; 8];
                            for (i, val) in in_vals.iter_mut().enumerate() {
                                let iw = (ow_base + i) * stride_w + kx;
                                if iw >= pad_w && iw - pad_w < w {
                                    *val = input.data[in_offset + ih * w + (iw - pad_w)];
                                }
                            }
                            let inv = f32x8::new(in_vals);
                            acc += inv * w_v;
                        }
                    }

                    let arr: [f32; 8] = acc.into();
                    output.data
                        [out_offset + oh * out_w + ow_base..out_offset + oh * out_w + ow_base + 8]
                        .copy_from_slice(&arr);
                }

                // Scalar remainder
                for ow in (chunks * 8)..out_w {
                    let mut sum = b;
                    for ky in 0..kh {
                        let ih = oh * stride_h + ky;
                        if ih < pad_h || ih - pad_h >= h {
                            continue;
                        }
                        let ih = ih - pad_h;
                        for kx in 0..kw {
                            let iw = ow * stride_w + kx;
                            if iw >= pad_w && iw - pad_w < w {
                                sum += input.data[in_offset + ih * w + (iw - pad_w)]
                                    * weight.data[w_offset + ky * kw + kx];
                            }
                        }
                    }
                    output.data[out_offset + oh * out_w + ow] = sum;
                }
            }
        }
    }
    output
}

/// im2col: unfold input patches into columns for GEMM-based convolution.
/// col layout: (Cin*KH*KW) rows x (out_h*out_w) columns, row-major.
#[allow(clippy::too_many_arguments)]
fn im2col(
    input: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    col: &mut [f32],
    out_h: usize,
    out_w: usize,
) {
    let out_spatial = out_h * out_w;
    for ci in 0..c {
        for ky in 0..kh {
            for kx in 0..kw {
                let row = (ci * kh + ky) * kw + kx;
                let row_offset = row * out_spatial;
                for oh in 0..out_h {
                    let ih = oh * sh + ky;
                    for ow in 0..out_w {
                        let iw = ow * sw + kx;
                        let val = if ih >= ph && ih - ph < h && iw >= pw && iw - pw < w {
                            input[ci * h * w + (ih - ph) * w + (iw - pw)]
                        } else {
                            0.0
                        };
                        col[row_offset + oh * out_w + ow] = val;
                    }
                }
            }
        }
    }
}

/// General convolution via im2col + GEMM
/// input (N,Cin,H,W), weight (Cout,Cin,KH,KW), bias (Cout,)
pub fn conv2d_general(
    input: &NdTensor<f32>,
    weight: &NdTensor<f32>,
    bias: Option<&NdTensor<f32>>,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> NdTensor<f32> {
    assert_eq!(input.ndim(), 4);
    assert_eq!(weight.ndim(), 4);
    let (n, cin, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let (cout, kh, kw) = (weight.shape[0], weight.shape[2], weight.shape[3]);
    assert_eq!(weight.shape[1], cin);

    let out_h = (h + 2 * pad_h - kh) / stride_h + 1;
    let out_w = (w + 2 * pad_w - kw) / stride_w + 1;
    let out_spatial = out_h * out_w;
    let k = cin * kh * kw;

    let mut output = NdTensor::zeros(&[n, cout, out_h, out_w]);
    let mut col = vec![0.0f32; k * out_spatial];

    for batch in 0..n {
        let in_offset = batch * cin * h * w;
        im2col(
            &input.data[in_offset..in_offset + cin * h * w],
            cin,
            h,
            w,
            kh,
            kw,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            &mut col,
            out_h,
            out_w,
        );

        // GEMM: weight (Cout, K) x col (K, out_spatial) = out (Cout, out_spatial)
        for co in 0..cout {
            let b = bias.map_or(0.0, |b| b.data[co]);
            let w_row = &weight.data[co * k..(co + 1) * k];
            let out_row_offset = (batch * cout + co) * out_spatial;

            for j in 0..out_spatial {
                let mut sum = b;
                // Dot product with SIMD
                let chunks = k / 8;
                for i in 0..chunks {
                    let o = i * 8;
                    let wv = f32x8::from(&w_row[o..o + 8]);
                    // col is (K, out_spatial) row-major: gather from strided memory
                    let mut cv = [0.0f32; 8];
                    for s in 0..8 {
                        cv[s] = col[(o + s) * out_spatial + j];
                    }
                    let colv = f32x8::new(cv);
                    let prod = wv * colv;
                    let arr: [f32; 8] = prod.into();
                    sum += arr.iter().sum::<f32>();
                }
                for i in (chunks * 8)..k {
                    sum += w_row[i] * col[i * out_spatial + j];
                }
                output.data[out_row_offset + j] = sum;
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_pointwise() {
        // input (1,2,2,3), weight (3,2,1,1), no bias
        let input = NdTensor::from_vec((0..12).map(|x| x as f32).collect(), &[1, 2, 2, 3]);
        // weight: 3 output channels, 2 input channels
        let weight = NdTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2, 1, 1]);
        let out = conv2d_pointwise(&input, &weight, None);
        assert_eq!(out.shape, vec![1, 3, 2, 3]);
        // Channel 0: copy input channel 0
        assert_eq!(out.data[0], 0.0);
        assert_eq!(out.data[1], 1.0);
        // Channel 1: copy input channel 1
        assert_eq!(out.data[6], 6.0);
        // Channel 2: sum of channels 0 and 1
        assert_eq!(out.data[12], 0.0 + 6.0);
        assert_eq!(out.data[13], 1.0 + 7.0);
    }

    #[test]
    fn test_conv2d_pointwise_with_bias() {
        let input = NdTensor::from_vec(vec![1.0; 4], &[1, 1, 2, 2]);
        let weight = NdTensor::from_vec(vec![2.0], &[1, 1, 1, 1]);
        let bias = NdTensor::from_vec(vec![0.5], &[1]);
        let out = conv2d_pointwise(&input, &weight, Some(&bias));
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        for &v in &out.data {
            assert!((v - 2.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_conv2d_depthwise_no_pad() {
        // input (1,1,3,3), weight (1,1,2,2), stride 1, pad 0
        let input = NdTensor::from_vec((0..9).map(|x| x as f32).collect(), &[1, 1, 3, 3]);
        let weight = NdTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let out = conv2d_depthwise(&input, &weight, None, 1, 1, 0, 0);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        // (0,0): 0*1 + 1*0 + 3*0 + 4*1 = 4
        assert!((out.data[0] - 4.0).abs() < 1e-6);
        // (0,1): 1*1 + 2*0 + 4*0 + 5*1 = 6
        assert!((out.data[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_conv2d_depthwise_with_pad() {
        // input (1,1,2,2), weight (1,1,3,3), stride 1, pad 1 -> out 2x2
        let input = NdTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let mut weight = NdTensor::from_vec(vec![0.0; 9], &[1, 1, 3, 3]);
        weight.data[4] = 1.0; // center of 3x3 = identity filter
        let out = conv2d_depthwise(&input, &weight, None, 1, 1, 1, 1);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        assert!((out.data[0] - 1.0).abs() < 1e-6);
        assert!((out.data[1] - 2.0).abs() < 1e-6);
        assert!((out.data[2] - 3.0).abs() < 1e-6);
        assert!((out.data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_conv2d_general_identity() {
        // 1x1 conv via general should match pointwise
        let input = NdTensor::from_vec((0..12).map(|x| x as f32).collect(), &[1, 3, 2, 2]);
        let weight = NdTensor::from_vec(
            vec![
                1.0, 0.0, 0.0, // out_ch 0: copy in_ch 0
                0.0, 1.0, 0.0, // out_ch 1: copy in_ch 1
            ],
            &[2, 3, 1, 1],
        );
        let out = conv2d_general(&input, &weight, None, 1, 1, 0, 0);
        assert_eq!(out.shape, vec![1, 2, 2, 2]);
        assert!((out.data[0] - 0.0).abs() < 1e-6);
        assert!((out.data[1] - 1.0).abs() < 1e-6);
        assert!((out.data[4] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_conv2d_general_3x3() {
        // input (1,1,4,4), weight (1,1,3,3) all ones, stride 1, pad 0 -> out 2x2
        let input = NdTensor::from_vec(vec![1.0; 16], &[1, 1, 4, 4]);
        let weight = NdTensor::from_vec(vec![1.0; 9], &[1, 1, 3, 3]);
        let out = conv2d_general(&input, &weight, None, 1, 1, 0, 0);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        for &v in &out.data {
            assert!((v - 9.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_conv2d_general_with_bias() {
        let input = NdTensor::from_vec(vec![1.0; 4], &[1, 1, 2, 2]);
        let weight = NdTensor::from_vec(vec![1.0], &[1, 1, 1, 1]);
        let bias = NdTensor::from_vec(vec![10.0], &[1]);
        let out = conv2d_general(&input, &weight, Some(&bias), 1, 1, 0, 0);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        for &v in &out.data {
            assert!((v - 11.0).abs() < 1e-6);
        }
    }
}
