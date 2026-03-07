use ndarray::Array2;
use ocrus_core::BBox;

/// Text orientation detected in the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextOrientation {
    Horizontal,
    Vertical,
    Mixed,
}

/// Compute the "sharpness" of a projection profile as variance / mean^2.
/// Higher sharpness means clearer peaks and valleys (stronger text structure).
fn profile_sharpness(profile: &[f64]) -> f64 {
    if profile.is_empty() {
        return 0.0;
    }
    let n = profile.len() as f64;
    let mean = profile.iter().sum::<f64>() / n;
    if mean < 1e-9 {
        return 0.0;
    }
    let variance = profile.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    variance / (mean * mean)
}

/// Detect text orientation by comparing horizontal and vertical projection profiles.
/// Returns the dominant orientation.
pub fn detect_orientation(binary: &Array2<u8>) -> TextOrientation {
    let (h, w) = binary.dim();
    if h == 0 || w == 0 {
        return TextOrientation::Horizontal;
    }

    // Horizontal projection: count foreground pixels per row
    let h_profile: Vec<f64> = (0..h)
        .map(|r| binary.row(r).iter().filter(|&&px| px == 0).count() as f64)
        .collect();

    // Vertical projection: count foreground pixels per column
    let v_profile: Vec<f64> = (0..w)
        .map(|c| binary.column(c).iter().filter(|&&px| px == 0).count() as f64)
        .collect();

    let h_sharpness = profile_sharpness(&h_profile);
    let v_sharpness = profile_sharpness(&v_profile);

    if v_sharpness < 1e-9 && h_sharpness < 1e-9 {
        return TextOrientation::Horizontal;
    }

    let ratio = if v_sharpness < 1e-9 {
        f64::INFINITY
    } else {
        h_sharpness / v_sharpness
    };

    if ratio > 1.5 {
        TextOrientation::Horizontal
    } else if ratio < 0.67 {
        TextOrientation::Vertical
    } else {
        TextOrientation::Mixed
    }
}

/// Detect vertical text columns using vertical projection profile.
/// Input: binary image (0=foreground/text, 255=background).
/// Returns bounding boxes for each detected column (right-to-left order for Japanese).
pub fn detect_columns_vertical(binary: &Array2<u8>) -> Vec<BBox> {
    let (h, w) = binary.dim();
    if h == 0 || w == 0 {
        return Vec::new();
    }

    let min_fg_count = (h as f64 * 0.01).max(1.0) as usize;
    let min_col_width = (w / 50).max(2);

    // Count foreground pixels per column
    let col_counts: Vec<usize> = (0..w)
        .map(|c| binary.column(c).iter().filter(|&&px| px == 0).count())
        .collect();

    // Group consecutive foreground columns
    let mut columns: Vec<(usize, usize)> = Vec::new();
    let mut start: Option<usize> = None;

    for (x, &count) in col_counts.iter().enumerate() {
        if count >= min_fg_count {
            if start.is_none() {
                start = Some(x);
            }
        } else if let Some(s) = start.take() {
            let width = x - s;
            if width >= min_col_width {
                columns.push((s, width));
            }
        }
    }
    // Handle trailing group
    if let Some(s) = start {
        let width = w - s;
        if width >= min_col_width {
            columns.push((s, width));
        }
    }

    // Build BBoxes and sort right-to-left (descending x) for Japanese reading order
    let mut bboxes: Vec<BBox> = columns
        .into_iter()
        .map(|(col_start, col_width)| BBox::new(col_start as u32, 0, col_width as u32, h as u32))
        .collect();

    bboxes.sort_by(|a, b| b.x.cmp(&a.x));
    bboxes
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn empty_image_returns_horizontal_and_no_columns() {
        let img = Array2::<u8>::from_elem((0, 0), 255);
        assert_eq!(detect_orientation(&img), TextOrientation::Horizontal);
        assert!(detect_columns_vertical(&img).is_empty());
    }

    #[test]
    fn all_white_returns_horizontal_and_no_columns() {
        let img = Array2::<u8>::from_elem((100, 100), 255);
        assert_eq!(detect_orientation(&img), TextOrientation::Horizontal);
        assert!(detect_columns_vertical(&img).is_empty());
    }

    #[test]
    fn horizontal_bars_detected_as_horizontal() {
        // Two horizontal bars (rows of black pixels)
        let mut img = Array2::<u8>::from_elem((100, 200), 255);
        // Bar 1: rows 20..30
        for r in 20..30 {
            for c in 10..190 {
                img[[r, c]] = 0;
            }
        }
        // Bar 2: rows 60..70
        for r in 60..70 {
            for c in 10..190 {
                img[[r, c]] = 0;
            }
        }

        assert_eq!(detect_orientation(&img), TextOrientation::Horizontal);
        // Horizontal bars should not produce meaningful vertical columns
        // (they span most of the width, forming one large group, not separate columns)
    }

    #[test]
    fn vertical_bars_detected_as_vertical() {
        // Three vertical bars (columns of black pixels)
        let mut img = Array2::<u8>::from_elem((200, 200), 255);
        // Bar 1: cols 20..35
        for r in 10..190 {
            for c in 20..35 {
                img[[r, c]] = 0;
            }
        }
        // Bar 2: cols 80..95
        for r in 10..190 {
            for c in 80..95 {
                img[[r, c]] = 0;
            }
        }
        // Bar 3: cols 140..155
        for r in 10..190 {
            for c in 140..155 {
                img[[r, c]] = 0;
            }
        }

        assert_eq!(detect_orientation(&img), TextOrientation::Vertical);

        let cols = detect_columns_vertical(&img);
        assert_eq!(cols.len(), 3);
        // Right-to-left order
        assert!(cols[0].x > cols[1].x);
        assert!(cols[1].x > cols[2].x);
    }

    #[test]
    fn columns_sorted_right_to_left() {
        let mut img = Array2::<u8>::from_elem((100, 100), 255);
        // Column at x=10..15
        for r in 0..100 {
            for c in 10..15 {
                img[[r, c]] = 0;
            }
        }
        // Column at x=50..55
        for r in 0..100 {
            for c in 50..55 {
                img[[r, c]] = 0;
            }
        }

        let cols = detect_columns_vertical(&img);
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].x, 50);
        assert_eq!(cols[1].x, 10);
    }
}
