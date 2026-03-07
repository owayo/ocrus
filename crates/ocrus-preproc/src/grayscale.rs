use image::DynamicImage;
use ndarray::Array2;
use wide::u16x8;

/// Convert an image to grayscale as a 2D ndarray (height x width), values in [0, 255].
pub fn to_grayscale(img: &DynamicImage) -> Array2<u8> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let raw = rgb.as_raw();
    let pixel_count = (w * h) as usize;

    let mut out = vec![0u8; pixel_count];
    grayscale_simd(raw, &mut out);

    Array2::from_shape_vec((h as usize, w as usize), out).expect("shape mismatch in to_grayscale")
}

/// SIMD grayscale: process 8 pixels at a time using integer BT.601 coefficients.
/// Formula: gray = (77*R + 150*G + 29*B) >> 8
fn grayscale_simd(rgb: &[u8], out: &mut [u8]) {
    let coeff_r = u16x8::splat(77);
    let coeff_g = u16x8::splat(150);
    let coeff_b = u16x8::splat(29);

    let chunks = out.len() / 8;
    let remainder = out.len() % 8;

    for i in 0..chunks {
        let base = i * 8;
        let rgb_base = base * 3;

        let mut r = [0u16; 8];
        let mut g = [0u16; 8];
        let mut b = [0u16; 8];

        for j in 0..8 {
            let idx = rgb_base + j * 3;
            r[j] = rgb[idx] as u16;
            g[j] = rgb[idx + 1] as u16;
            b[j] = rgb[idx + 2] as u16;
        }

        let vr = u16x8::new(r);
        let vg = u16x8::new(g);
        let vb = u16x8::new(b);

        let gray16: u16x8 = (vr * coeff_r + vg * coeff_g + vb * coeff_b) >> 8;
        let arr: [u16; 8] = gray16.to_array();

        for j in 0..8 {
            out[base + j] = arr[j] as u8;
        }
    }

    // Scalar fallback for remaining pixels
    let start = chunks * 8;
    for i in 0..remainder {
        let idx = (start + i) * 3;
        let r = rgb[idx] as u16;
        let g = rgb[idx + 1] as u16;
        let b = rgb[idx + 2] as u16;
        out[start + i] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    #[test]
    fn test_to_grayscale_shape() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 50));
        let gray = to_grayscale(&img);
        assert_eq!(gray.shape(), &[50, 100]);
    }

    #[test]
    fn test_grayscale_simd_accuracy() {
        // Test that SIMD matches scalar BT.601 for known values
        let rgb = vec![
            255, 0, 0, // pure red
            0, 255, 0, // pure green
            0, 0, 255, // pure blue
            128, 128, 128, // mid gray
            255, 255, 255, // white
            0, 0, 0, // black
            200, 100, 50, // mixed
            10, 20, 30, // dark mixed
        ];
        let mut out = vec![0u8; 8];
        grayscale_simd(&rgb, &mut out);

        // Expected: (77*R + 150*G + 29*B) >> 8
        let expected: Vec<u8> = (0..8)
            .map(|i| {
                let r = rgb[i * 3] as u16;
                let g = rgb[i * 3 + 1] as u16;
                let b = rgb[i * 3 + 2] as u16;
                ((77 * r + 150 * g + 29 * b) >> 8) as u8
            })
            .collect();

        assert_eq!(out, expected);
    }

    #[test]
    fn test_grayscale_simd_remainder() {
        // 11 pixels: 8 SIMD + 3 remainder
        let mut rgb = Vec::new();
        for i in 0..11u8 {
            rgb.push(i * 20);
            rgb.push(i * 10);
            rgb.push(i * 5);
        }
        let mut out = vec![0u8; 11];
        grayscale_simd(&rgb, &mut out);

        for i in 0..11 {
            let r = rgb[i * 3] as u16;
            let g = rgb[i * 3 + 1] as u16;
            let b = rgb[i * 3 + 2] as u16;
            let expected = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
            assert_eq!(out[i], expected, "mismatch at pixel {i}");
        }
    }
}
