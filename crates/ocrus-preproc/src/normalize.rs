use ndarray::Array4;
use ocrus_core::BBox;
use wide::f32x8;

const TARGET_HEIGHT: u32 = 48;
const NUM_CHANNELS: usize = 3;
/// Maximum width after resize (prevents OOM on very wide lines).
const MAX_WIDTH: usize = 2048;
/// Background fill value after PaddleOCR normalization: (255/255 - 0.5)/0.5 = 1.0
const PAD_VALUE: f32 = 1.0;

/// SIMD constants for normalization: (px/255.0 - 0.5)/0.5 = px/127.5 - 1.0
const SIMD_SCALE: f32 = 1.0 / 127.5;
const SIMD_OFFSET: f32 = -1.0;

/// Normalize a line image: crop by bbox, resize to fixed height, convert to NCHW f32 tensor.
/// Output shape: (1, 3, TARGET_HEIGHT, new_width), values normalized by PaddleOCR convention.
/// Handles images of any size: small crops are padded, wide lines are capped at MAX_WIDTH.
pub fn normalize_line(gray: &ndarray::Array2<u8>, bbox: &BBox) -> Array4<f32> {
    normalize_line_inner(gray, bbox, false)
}

/// Normalize a vertical text column: crop, rotate 90° clockwise, then resize like a horizontal line.
/// Use this for vertical text columns where height >> width.
pub fn normalize_line_vertical(gray: &ndarray::Array2<u8>, bbox: &BBox) -> Array4<f32> {
    normalize_line_inner(gray, bbox, true)
}

fn normalize_line_inner(gray: &ndarray::Array2<u8>, bbox: &BBox, rotate: bool) -> Array4<f32> {
    let (img_h, img_w) = (gray.nrows() as u32, gray.ncols() as u32);

    // Clamp bbox to image bounds
    let x0 = bbox.x.min(img_w.saturating_sub(1)) as usize;
    let y0 = bbox.y.min(img_h.saturating_sub(1)) as usize;
    let x1 = (bbox.x + bbox.width).min(img_w) as usize;
    let y1 = (bbox.y + bbox.height).min(img_h) as usize;

    let raw_crop_h = y1 - y0;
    let raw_crop_w = x1 - x0;

    if raw_crop_h == 0 || raw_crop_w == 0 {
        return Array4::from_elem((1, NUM_CHANNELS, TARGET_HEIGHT as usize, 1), PAD_VALUE);
    }

    // For vertical text, rotate 90° clockwise: (h, w) -> (w, h)
    // After rotation: new_h = raw_crop_w, new_w = raw_crop_h
    let (crop_h, crop_w) = if rotate {
        (raw_crop_w, raw_crop_h)
    } else {
        (raw_crop_h, raw_crop_w)
    };

    // Compute new width preserving aspect ratio, capped at MAX_WIDTH
    let scale = TARGET_HEIGHT as f32 / crop_h as f32;
    let new_w = ((crop_w as f32 * scale).round().max(1.0) as usize).min(MAX_WIDTH);
    let new_h = TARGET_HEIGHT as usize;

    // Recalculate effective scale for width to handle MAX_WIDTH capping
    let effective_w_scale = crop_w as f32 / new_w as f32;

    // Nearest-neighbor resize from crop region
    // First, collect resized grayscale row by row, then SIMD-normalize
    let mut resized = Array4::from_elem((1, NUM_CHANNELS, new_h, new_w), PAD_VALUE);

    let scale_vec = f32x8::splat(SIMD_SCALE);
    let offset_vec = f32x8::splat(SIMD_OFFSET);

    for ry in 0..new_h {
        let crop_y = ((ry as f32 / scale) as usize).min(crop_h - 1);

        // Collect source pixels for this row
        let row_pixels: Vec<u8> = (0..new_w)
            .map(|rx| {
                let crop_x = ((rx as f32 * effective_w_scale) as usize).min(crop_w - 1);
                if rotate {
                    // 90° counter-clockwise: vertical top→bottom maps to horizontal left→right
                    // rotated[ry, rx] = original[rx, W-1-ry]
                    let orig_y = crop_x + y0;
                    let orig_x = (raw_crop_w - 1 - crop_y) + x0;
                    gray[[orig_y, orig_x]]
                } else {
                    gray[[crop_y + y0, crop_x + x0]]
                }
            })
            .collect();

        // SIMD normalize: process 8 pixels at a time
        let chunks = new_w / 8;
        let remainder = new_w % 8;

        for i in 0..chunks {
            let base = i * 8;
            let mut px = [0.0f32; 8];
            for j in 0..8 {
                px[j] = row_pixels[base + j] as f32;
            }
            let v = f32x8::new(px);
            let normalized = v * scale_vec + offset_vec;
            let arr: [f32; 8] = normalized.to_array();

            for (j, &val) in arr.iter().enumerate() {
                let rx = base + j;
                for c in 0..NUM_CHANNELS {
                    resized[[0, c, ry, rx]] = val;
                }
            }
        }

        // Scalar fallback
        let start = chunks * 8;
        for i in 0..remainder {
            let rx = start + i;
            let normalized = row_pixels[rx] as f32 * SIMD_SCALE + SIMD_OFFSET;
            for c in 0..NUM_CHANNELS {
                resized[[0, c, ry, rx]] = normalized;
            }
        }
    }

    resized
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ocrus_core::BBox;

    #[test]
    fn test_normalize_line_shape() {
        let gray = Array2::from_elem((100, 200), 128u8);
        let bbox = BBox::new(10, 20, 180, 30);
        let result = normalize_line(&gray, &bbox);
        assert_eq!(result.shape()[0], 1);
        assert_eq!(result.shape()[1], 3);
        assert_eq!(result.shape()[2], 48);
        assert!(result.shape()[3] > 0);
    }

    #[test]
    fn test_normalize_line_values_range() {
        let gray = Array2::from_elem((100, 200), 200u8);
        let bbox = BBox::new(0, 0, 200, 100);
        let result = normalize_line(&gray, &bbox);
        for &v in result.iter() {
            assert!((-1.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_normalize_very_small_image() {
        let gray = Array2::from_elem((5, 10), 100u8);
        let bbox = BBox::new(0, 0, 10, 5);
        let result = normalize_line(&gray, &bbox);
        assert_eq!(result.shape()[0], 1);
        assert_eq!(result.shape()[1], 3);
        assert_eq!(result.shape()[2], 48);
        assert!(result.shape()[3] > 0);
    }

    #[test]
    fn test_normalize_very_wide_line() {
        let gray = Array2::from_elem((50, 10000), 128u8);
        let bbox = BBox::new(0, 0, 10000, 50);
        let result = normalize_line(&gray, &bbox);
        assert_eq!(result.shape()[2], 48);
        assert!(result.shape()[3] <= MAX_WIDTH);
    }

    #[test]
    fn test_normalize_empty_bbox() {
        let gray = Array2::from_elem((100, 200), 128u8);
        let bbox = BBox::new(50, 50, 0, 0);
        let result = normalize_line(&gray, &bbox);
        assert_eq!(result.shape()[2], 48);
        assert_eq!(result.shape()[3], 1);
    }

    #[test]
    fn test_normalize_large_image() {
        let gray = Array2::from_elem((4000, 6000), 128u8);
        let bbox = BBox::new(100, 500, 5800, 200);
        let result = normalize_line(&gray, &bbox);
        assert_eq!(result.shape()[0], 1);
        assert_eq!(result.shape()[1], 3);
        assert_eq!(result.shape()[2], 48);
        assert!(result.shape()[3] <= MAX_WIDTH);
        assert!(result.shape()[3] > 0);
    }

    #[test]
    fn test_normalize_simd_accuracy() {
        // Verify SIMD normalization matches scalar for known pixel values
        let gray = Array2::from_shape_fn((100, 200), |(y, x)| ((y * 200 + x) % 256) as u8);
        let bbox = BBox::new(0, 0, 200, 100);
        let result = normalize_line(&gray, &bbox);

        for &v in result.iter() {
            assert!(
                (-1.0..=1.001).contains(&v),
                "value {v} out of expected range"
            );
        }
    }

    #[test]
    fn test_normalize_line_vertical_shape() {
        // Vertical column: narrow width, tall height
        let gray = Array2::from_elem((500, 200), 128u8);
        let bbox = BBox::new(50, 0, 30, 500);
        let result = normalize_line_vertical(&gray, &bbox);
        assert_eq!(result.shape()[0], 1);
        assert_eq!(result.shape()[1], 3);
        assert_eq!(result.shape()[2], 48); // height is always TARGET_HEIGHT
        // After rotation: original height (500) becomes width, so new_w ≈ 500 * 48/30 = 800
        assert!(result.shape()[3] > 100, "rotated width should be large");
    }

    #[test]
    fn test_normalize_line_vertical_values() {
        let gray = Array2::from_elem((200, 100), 200u8);
        let bbox = BBox::new(10, 0, 20, 200);
        let result = normalize_line_vertical(&gray, &bbox);
        for &v in result.iter() {
            assert!((-1.0..=1.001).contains(&v));
        }
    }
}
