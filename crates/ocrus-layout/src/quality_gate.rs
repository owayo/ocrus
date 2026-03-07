use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Image quality assessment for adaptive pipeline selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageQuality {
    /// Contrast ratio between foreground and background (0.0 - 1.0)
    pub contrast: f32,
    /// Binarization quality score (0.0 - 1.0)
    pub binarization_quality: f32,
    /// Estimated skew angle in degrees
    pub skew_angle: f32,
}

/// Assess the quality of a binarized image.
///
/// Input: binary image where 0 = foreground (text), 255 = background.
pub fn assess_quality(binary: &Array2<u8>) -> ImageQuality {
    let contrast = compute_contrast(binary);
    let binarization_quality = compute_binarization_quality(binary);
    let skew_angle = estimate_skew(binary);

    ImageQuality {
        contrast,
        binarization_quality,
        skew_angle,
    }
}

/// Determine if the fast path can be used based on quality.
pub fn should_use_fast_path(quality: &ImageQuality) -> bool {
    quality.contrast > 0.3 && quality.binarization_quality > 0.5 && quality.skew_angle.abs() < 2.0
}

/// Compute contrast as the ratio of foreground pixels.
/// Good contrast means a reasonable balance of fg/bg (not too sparse, not too dense).
/// Returns a score in [0.0, 1.0] where higher is better.
fn compute_contrast(binary: &Array2<u8>) -> f32 {
    let total = binary.len();
    if total == 0 {
        return 0.0;
    }

    let fg_count = binary.iter().filter(|&&v| v == 0).count();
    let fg_ratio = fg_count as f32 / total as f32;

    // Ideal foreground ratio is around 5-30% for text documents.
    // Score peaks at ~15% and drops off symmetrically.
    let ideal = 0.15;
    let deviation = (fg_ratio - ideal).abs();
    (1.0 - (deviation / ideal).min(1.0)).max(0.0)
}

/// Compute binarization quality from line height consistency.
/// Uses horizontal projection profile to find text lines, then measures
/// variance of line heights. Low variance = high quality.
fn compute_binarization_quality(binary: &Array2<u8>) -> f32 {
    let (h, w) = (binary.nrows(), binary.ncols());
    if h == 0 || w == 0 {
        return 0.0;
    }

    // Horizontal projection profile
    let projection: Vec<u32> = (0..h)
        .map(|y| binary.row(y).iter().filter(|&&v| v == 0).count() as u32)
        .collect();

    let min_fg = (w as f32 * 0.01).max(1.0) as u32;
    let min_line_height = (h / 50).max(2);

    // Find line heights
    let mut line_heights: Vec<f32> = Vec::new();
    let mut in_line = false;
    let mut line_start = 0;

    for (y, &count) in projection.iter().enumerate() {
        if count >= min_fg {
            if !in_line {
                in_line = true;
                line_start = y;
            }
        } else if in_line {
            in_line = false;
            let lh = y - line_start;
            if lh >= min_line_height {
                line_heights.push(lh as f32);
            }
        }
    }
    if in_line {
        let lh = h - line_start;
        if lh >= min_line_height {
            line_heights.push(lh as f32);
        }
    }

    if line_heights.len() < 2 {
        // Single line or no lines: assume decent quality
        return if line_heights.is_empty() { 0.0 } else { 0.8 };
    }

    let mean = line_heights.iter().sum::<f32>() / line_heights.len() as f32;
    if mean == 0.0 {
        return 0.0;
    }

    let variance = line_heights
        .iter()
        .map(|&h| (h - mean).powi(2))
        .sum::<f32>()
        / line_heights.len() as f32;
    let cv = variance.sqrt() / mean; // coefficient of variation

    // CV close to 0 = very consistent = high quality
    // CV > 0.5 = poor quality
    (1.0 - (cv / 0.5).min(1.0)).max(0.0)
}

/// Estimate skew angle using projection profile offsets.
/// Computes projection profiles at small angular offsets and finds the angle
/// that maximizes the peak-to-valley ratio (sharpest projection = correct angle).
fn estimate_skew(binary: &Array2<u8>) -> f32 {
    let (h, w) = (binary.nrows(), binary.ncols());
    if h < 10 || w < 10 {
        return 0.0;
    }

    let mut best_angle = 0.0f32;
    let mut best_score = f32::MIN;

    // Test angles from -5 to +5 degrees in 0.5 degree steps
    let mut angle = -5.0f32;
    while angle <= 5.0 {
        let score = projection_sharpness(binary, h, w, angle);
        if score > best_score {
            best_score = score;
            best_angle = angle;
        }
        angle += 0.5;
    }

    best_angle
}

/// Compute the sharpness of horizontal projection at a given angle.
/// Sharpness = variance of the projection profile (higher = sharper peaks).
fn projection_sharpness(binary: &Array2<u8>, h: usize, w: usize, angle_deg: f32) -> f32 {
    let angle_rad = angle_deg * std::f32::consts::PI / 180.0;
    let tan = angle_rad.tan();

    let mut projection = vec![0u32; h];

    for (y, proj) in projection.iter_mut().enumerate() {
        let mut count = 0u32;
        for x in 0..w {
            let offset = (x as f32 * tan) as i32;
            let mapped_y = y as i32 + offset;
            if mapped_y >= 0 && (mapped_y as usize) < h && binary[[mapped_y as usize, x]] == 0 {
                count += 1;
            }
        }
        *proj = count;
    }

    // Compute variance of projection
    let mean = projection.iter().sum::<u32>() as f32 / h as f32;
    projection
        .iter()
        .map(|&v| (v as f32 - mean).powi(2))
        .sum::<f32>()
        / h as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_assess_quality_blank_image() {
        let img = Array2::from_elem((100, 200), 255u8);
        let q = assess_quality(&img);
        assert!(q.contrast < 0.1, "blank image should have low contrast");
        assert_eq!(q.binarization_quality, 0.0);
    }

    #[test]
    fn test_assess_quality_good_text() {
        let mut img = Array2::from_elem((200, 400), 255u8);
        // Draw two consistent text lines (30px each)
        for y in 30..60 {
            for x in 20..380 {
                img[[y, x]] = 0;
            }
        }
        for y in 100..130 {
            for x in 20..380 {
                img[[y, x]] = 0;
            }
        }

        let q = assess_quality(&img);
        assert!(
            q.binarization_quality > 0.8,
            "consistent lines should have high binarization quality: {}",
            q.binarization_quality
        );
        assert!(
            q.skew_angle.abs() < 2.0,
            "horizontal lines should have near-zero skew: {}",
            q.skew_angle
        );
    }

    #[test]
    fn test_should_use_fast_path() {
        let good = ImageQuality {
            contrast: 0.8,
            binarization_quality: 0.9,
            skew_angle: 0.5,
        };
        assert!(should_use_fast_path(&good));

        let bad_contrast = ImageQuality {
            contrast: 0.1,
            binarization_quality: 0.9,
            skew_angle: 0.5,
        };
        assert!(!should_use_fast_path(&bad_contrast));

        let bad_skew = ImageQuality {
            contrast: 0.8,
            binarization_quality: 0.9,
            skew_angle: 5.0,
        };
        assert!(!should_use_fast_path(&bad_skew));
    }

    #[test]
    fn test_empty_image() {
        let img = Array2::from_elem((0, 0), 255u8);
        let q = assess_quality(&img);
        assert_eq!(q.contrast, 0.0);
        assert_eq!(q.binarization_quality, 0.0);
        assert_eq!(q.skew_angle, 0.0);
    }
}
