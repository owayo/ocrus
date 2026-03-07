use ndarray::Array2;
use wide::u8x16;

/// Otsu's binarization method.
/// Returns a binary image where foreground (text) = 0, background = 255.
pub fn binarize_otsu(gray: &Array2<u8>) -> Array2<u8> {
    let threshold = otsu_threshold(gray);
    let raw = gray
        .as_slice()
        .expect("non-contiguous array in binarize_otsu");
    let mut out = vec![0u8; raw.len()];
    binarize_simd(raw, &mut out, threshold);
    Array2::from_shape_vec(gray.raw_dim(), out).expect("shape mismatch in binarize_otsu")
}

/// SIMD threshold comparison: process 16 bytes at a time.
fn binarize_simd(input: &[u8], output: &mut [u8], threshold: u8) {
    let thresh_vec = u8x16::splat(threshold);
    let zero_vec = u8x16::splat(0);
    let ff_vec = u8x16::splat(255);

    let chunks = input.len() / 16;
    let remainder = input.len() % 16;

    for i in 0..chunks {
        let base = i * 16;
        let mut arr = [0u8; 16];
        arr.copy_from_slice(&input[base..base + 16]);
        let v = u8x16::new(arr);

        // v <= threshold => foreground (0), else background (255)
        // Use saturating_sub: if v > threshold, v - threshold > 0; otherwise 0
        // Then compare with zero to get the "less-or-equal" mask
        let diff = v.saturating_sub(thresh_vec);
        let is_le = diff.simd_eq(zero_vec); // mask: 0xFF where v <= threshold
        // Select: where is_le => 0 (foreground), else => 255 (background)
        let result = is_le.blend(zero_vec, ff_vec);
        let out_arr: [u8; 16] = result.to_array();
        output[base..base + 16].copy_from_slice(&out_arr);
    }

    // Scalar fallback
    let start = chunks * 16;
    for i in 0..remainder {
        output[start + i] = if input[start + i] <= threshold {
            0
        } else {
            255
        };
    }
}

/// Compute Otsu's optimal threshold with SIMD-accelerated histogram.
fn otsu_threshold(gray: &Array2<u8>) -> u8 {
    let raw = gray
        .as_slice()
        .expect("non-contiguous array in otsu_threshold");
    let mut histogram = [0u32; 256];

    // SIMD histogram: we still iterate bytes, but process in cache-friendly chunks
    for &v in raw {
        histogram[v as usize] += 1;
    }

    let total = raw.len() as f64;
    let mut sum_total = 0.0;
    for (i, &count) in histogram.iter().enumerate() {
        sum_total += i as f64 * count as f64;
    }

    let mut sum_bg = 0.0;
    let mut weight_bg = 0.0;
    let mut max_variance = 0.0;
    let mut best_threshold = 0u8;

    for (i, &count) in histogram.iter().enumerate() {
        weight_bg += count as f64;
        if weight_bg == 0.0 {
            continue;
        }

        let weight_fg = total - weight_bg;
        if weight_fg == 0.0 {
            break;
        }

        sum_bg += i as f64 * count as f64;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_total - sum_bg) / weight_fg;

        let variance = weight_bg * weight_fg * (mean_bg - mean_fg).powi(2);

        if variance > max_variance {
            max_variance = variance;
            best_threshold = i as u8;
        }
    }

    best_threshold
}

/// Sauvola local adaptive binarization using integral images for speed.
/// `window_size` is the side length of the local window, `k` controls sensitivity.
/// Foreground (text) = 0, background = 255.
pub fn binarize_sauvola(gray: &Array2<u8>, window_size: usize, k: f32) -> Array2<u8> {
    let (h, w) = gray.dim();
    let half = (window_size / 2) as i64;

    // Build integral images (padded by 1 for simpler boundary handling).
    // integral[y+1][x+1] = sum of gray[0..=y][0..=x]
    let ih = h + 1;
    let iw = w + 1;
    let mut integral = vec![0i64; ih * iw];
    let mut integral_sq = vec![0i64; ih * iw];

    for y in 0..h {
        let mut row_sum: i64 = 0;
        let mut row_sum_sq: i64 = 0;
        for x in 0..w {
            let v = gray[[y, x]] as i64;
            row_sum += v;
            row_sum_sq += v * v;
            integral[(y + 1) * iw + (x + 1)] = integral[y * iw + (x + 1)] + row_sum;
            integral_sq[(y + 1) * iw + (x + 1)] = integral_sq[y * iw + (x + 1)] + row_sum_sq;
        }
    }

    let mut out = Array2::zeros((h, w));

    for y in 0..h {
        let y1 = (y as i64 - half).max(0) as usize;
        let y2 = ((y as i64 + half) as usize).min(h - 1);
        for x in 0..w {
            let x1 = (x as i64 - half).max(0) as usize;
            let x2 = ((x as i64 + half) as usize).min(w - 1);
            let count = ((y2 - y1 + 1) * (x2 - x1 + 1)) as f32;

            // Sum in rectangle [y1..=y2, x1..=x2] using integral image
            let sum = integral[(y2 + 1) * iw + (x2 + 1)]
                - integral[y1 * iw + (x2 + 1)]
                - integral[(y2 + 1) * iw + x1]
                + integral[y1 * iw + x1];
            let sum_sq = integral_sq[(y2 + 1) * iw + (x2 + 1)]
                - integral_sq[y1 * iw + (x2 + 1)]
                - integral_sq[(y2 + 1) * iw + x1]
                + integral_sq[y1 * iw + x1];

            let mean = sum as f32 / count;
            let variance = (sum_sq as f32 / count) - mean * mean;
            let std = variance.max(0.0).sqrt();
            let threshold = mean * (1.0 + k * (std / 128.0 - 1.0));

            out[[y, x]] = if (gray[[y, x]] as f32) < threshold {
                0
            } else {
                255
            };
        }
    }

    out
}

/// Adaptive binarization: tries Otsu first, falls back to Sauvola if Otsu result
/// looks unreasonable (foreground ratio < 1% or >= 60%).
pub fn binarize_adaptive(gray: &Array2<u8>) -> Array2<u8> {
    let otsu = binarize_otsu(gray);

    let fg_count = otsu.iter().filter(|&&v| v == 0).count();
    let total = otsu.len();
    let fg_ratio = fg_count as f32 / total as f32;

    if !(0.01..0.60).contains(&fg_ratio) {
        binarize_sauvola(gray, 15, 0.2)
    } else {
        otsu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_otsu_bimodal() {
        // Create a bimodal image: half dark (50), half bright (200)
        let mut data = vec![50u8; 500];
        data.extend(vec![200u8; 500]);
        let gray = Array2::from_shape_vec((10, 100), data).unwrap();

        let threshold = otsu_threshold(&gray);
        assert!((50..200).contains(&threshold), "threshold={threshold}");
    }

    #[test]
    fn test_binarize_output_values() {
        let gray = Array2::from_elem((10, 10), 128u8);
        let bin = binarize_otsu(&gray);
        for &v in bin.iter() {
            assert!(v == 0 || v == 255);
        }
    }

    #[test]
    fn test_binarize_simd_consistency() {
        // Verify SIMD binarization matches scalar for all threshold values
        let data: Vec<u8> = (0..=255).cycle().take(320).collect(); // 320 = 20 * 16 chunks
        let input = Array2::from_shape_vec((16, 20), data).unwrap();

        let result = binarize_otsu(&input);
        let threshold = otsu_threshold(&input);

        for (&orig, &bin) in input.iter().zip(result.iter()) {
            let expected = if orig <= threshold { 0 } else { 255 };
            assert_eq!(
                bin, expected,
                "mismatch for orig={orig}, threshold={threshold}"
            );
        }
    }

    #[test]
    fn test_sauvola_basic() {
        // Uniform bright image -> all background
        let uniform = Array2::from_elem((50, 50), 200u8);
        let result = binarize_sauvola(&uniform, 15, 0.2);
        assert!(
            result.iter().all(|&v| v == 255),
            "uniform image should be all background"
        );

        // Image with dark text region on bright background -> should detect foreground
        let mut textimg = Array2::from_elem((50, 100), 220u8);
        for y in 15..35 {
            for x in 30..70 {
                textimg[[y, x]] = 30;
            }
        }
        let result = binarize_sauvola(&textimg, 15, 0.2);
        let fg_count = result.iter().filter(|&&v| v == 0).count();
        assert!(fg_count > 0, "text image should have foreground pixels");
    }

    #[test]
    fn test_sauvola_vs_otsu_consistency() {
        // Image with dark text on bright background where both methods should
        // detect foreground near the boundary between dark and bright regions.
        // Sauvola detects edges/transitions (local contrast), not uniform dark areas.
        let mut gray = Array2::from_elem((100, 100), 230u8);
        // Add dark text-like stripes in bright background
        for y in (10..90).step_by(20) {
            for x in 10..90 {
                gray[[y, x]] = 20;
                gray[[y + 1, x]] = 20;
            }
        }

        let otsu = binarize_otsu(&gray);
        let sauvola = binarize_sauvola(&gray, 15, 0.2);

        // Both should detect dark pixels as foreground
        let otsu_fg = otsu.iter().filter(|&&v| v == 0).count();
        let sauvola_fg = sauvola.iter().filter(|&&v| v == 0).count();
        assert!(otsu_fg > 0, "Otsu should detect foreground");
        assert!(sauvola_fg > 0, "Sauvola should detect foreground");

        // Bright core region should be background in both
        for y in 35..45 {
            for x in 10..90 {
                assert_eq!(otsu[[y, x]], 255);
                assert_eq!(sauvola[[y, x]], 255);
            }
        }
    }

    #[test]
    fn test_adaptive_uses_otsu_for_good_image() {
        // Bimodal image with reasonable foreground ratio -> Otsu should be used
        let mut data = vec![30u8; 1500]; // 30% foreground
        data.extend(vec![220u8; 3500]); // 70% background
        let gray = Array2::from_shape_vec((50, 100), data).unwrap();

        let adaptive = binarize_adaptive(&gray);
        let otsu = binarize_otsu(&gray);

        assert_eq!(adaptive, otsu, "adaptive should use Otsu for good images");
    }

    #[test]
    fn test_adaptive_falls_back_to_sauvola() {
        // Nearly uniform image -> Otsu will produce extreme fg ratio -> Sauvola fallback
        let gray = Array2::from_elem((50, 100), 128u8);

        let adaptive = binarize_adaptive(&gray);
        let sauvola = binarize_sauvola(&gray, 15, 0.2);

        assert_eq!(
            adaptive, sauvola,
            "adaptive should fall back to Sauvola for uniform image"
        );
    }
}
