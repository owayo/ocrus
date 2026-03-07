use ndarray::Array2;
use ocrus_core::BBox;
use wide::u8x16;

/// Detect text lines using horizontal projection profile.
/// Input: binary image (0=foreground/text, 255=background).
/// Returns bounding boxes for each detected line.
pub fn detect_lines_projection(binary: &Array2<u8>) -> Vec<BBox> {
    let (h, w) = (binary.nrows(), binary.ncols());
    if h == 0 || w == 0 {
        return Vec::new();
    }

    // Compute horizontal projection: count foreground pixels per row (SIMD-accelerated)
    let projection: Vec<u32> = (0..h)
        .map(|y| count_foreground_simd(binary.row(y).as_slice().unwrap()))
        .collect();

    // Find line boundaries: contiguous runs of rows with foreground pixels
    let min_fg_count = (w as f32 * 0.01).max(1.0) as u32;
    // Minimum line height scales with image size (at least 2px for tiny images)
    let min_line_height = (h / 50).max(2);
    let mut lines = Vec::new();
    let mut in_line = false;
    let mut line_start = 0;

    for (y, &count) in projection.iter().enumerate() {
        if count >= min_fg_count {
            if !in_line {
                in_line = true;
                line_start = y;
            }
        } else if in_line {
            in_line = false;
            let line_height = y - line_start;
            if line_height >= min_line_height {
                lines.push(BBox::new(
                    0,
                    line_start as u32,
                    w as u32,
                    line_height as u32,
                ));
            }
        }
    }

    // Handle line at bottom of image
    if in_line {
        let line_height = h - line_start;
        if line_height >= min_line_height {
            lines.push(BBox::new(
                0,
                line_start as u32,
                w as u32,
                line_height as u32,
            ));
        }
    }

    lines
}

/// Count foreground pixels (value == 0) in a row using SIMD.
/// Process 16 bytes at a time: compare with zero, count set bytes in mask.
fn count_foreground_simd(row: &[u8]) -> u32 {
    let zero_vec = u8x16::splat(0);
    let one_vec = u8x16::splat(1);
    let chunks = row.len() / 16;
    let remainder = row.len() % 16;
    let mut count: u32 = 0;

    for i in 0..chunks {
        let base = i * 16;
        let mut arr = [0u8; 16];
        arr.copy_from_slice(&row[base..base + 16]);
        let v = u8x16::new(arr);
        // cmp_eq returns mask: 0xFF where equal, 0x00 where not
        let eq_mask = v.simd_eq(zero_vec);
        // AND with 1 to get 0 or 1 per lane, then sum
        let ones = eq_mask & one_vec;
        let arr_out: [u8; 16] = ones.to_array();
        // Sum the 16 lanes
        let lane_sum: u32 = arr_out.iter().map(|&x| x as u32).sum();
        count += lane_sum;
    }

    // Scalar fallback
    let start = chunks * 16;
    for i in 0..remainder {
        if row[start + i] == 0 {
            count += 1;
        }
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_detect_lines_empty_image() {
        let img = Array2::from_elem((100, 200), 255u8);
        let lines = detect_lines_projection(&img);
        assert!(lines.is_empty());
    }

    #[test]
    fn test_detect_two_lines() {
        let mut img = Array2::from_elem((100, 200), 255u8);

        // Draw two horizontal text lines
        for y in 10..30 {
            for x in 10..190 {
                img[[y, x]] = 0;
            }
        }
        for y in 60..80 {
            for x in 10..190 {
                img[[y, x]] = 0;
            }
        }

        let lines = detect_lines_projection(&img);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].y < lines[1].y);
    }

    #[test]
    fn test_detect_lines_small_image() {
        // 20x50 tiny image with one line
        let mut img = Array2::from_elem((20, 50), 255u8);
        for y in 5..15 {
            for x in 2..48 {
                img[[y, x]] = 0;
            }
        }
        let lines = detect_lines_projection(&img);
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn test_detect_lines_large_image() {
        // 4000x6000 large image with 3 lines
        let mut img = Array2::from_elem((4000, 6000), 255u8);
        for y in 200..400 {
            for x in 100..5900 {
                img[[y, x]] = 0;
            }
        }
        for y in 1500..1700 {
            for x in 100..5900 {
                img[[y, x]] = 0;
            }
        }
        for y in 3000..3200 {
            for x in 100..5900 {
                img[[y, x]] = 0;
            }
        }
        let lines = detect_lines_projection(&img);
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_count_foreground_simd_accuracy() {
        // 32 bytes: 16 zeros + 16 non-zeros
        let mut row = vec![255u8; 32];
        for item in row.iter_mut().take(16) {
            *item = 0;
        }
        assert_eq!(count_foreground_simd(&row), 16);

        // All zeros
        let all_zero = vec![0u8; 50];
        assert_eq!(count_foreground_simd(&all_zero), 50);

        // No zeros
        let no_zero = vec![128u8; 50];
        assert_eq!(count_foreground_simd(&no_zero), 0);
    }
}
