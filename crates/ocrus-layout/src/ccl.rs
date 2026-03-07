use image::{GrayImage, Luma};
use imageproc::region_labelling::{Connectivity, connected_components};
use ndarray::Array2;
use ocrus_core::BBox;

/// Detect text lines using Connected Component Labeling.
/// Groups individual character components into lines based on vertical overlap.
/// Input: binary image (0=foreground/text, 255=background).
pub fn detect_lines_ccl(binary: &Array2<u8>) -> Vec<BBox> {
    let (h, w) = binary.dim();
    if h == 0 || w == 0 {
        return Vec::new();
    }

    // Convert ndarray to GrayImage
    let img = GrayImage::from_fn(w as u32, h as u32, |x, y| {
        Luma([binary[[y as usize, x as usize]]])
    });

    // Run connected components (background = 255 white)
    let labeled = connected_components(&img, Connectivity::Eight, Luma([255u8]));

    // Compute bounding box for each label
    let (iw, ih) = (labeled.width(), labeled.height());
    let mut label_bboxes: std::collections::HashMap<u32, (u32, u32, u32, u32)> =
        std::collections::HashMap::new();

    for y in 0..ih {
        for x in 0..iw {
            let label = labeled.get_pixel(x, y).0[0];
            if label == 0 {
                continue; // background
            }
            label_bboxes
                .entry(label)
                .and_modify(|(min_x, min_y, max_x, max_y)| {
                    *min_x = (*min_x).min(x);
                    *min_y = (*min_y).min(y);
                    *max_x = (*max_x).max(x);
                    *max_y = (*max_y).max(y);
                })
                .or_insert((x, y, x, y));
        }
    }

    // Filter noise: too small or too large components
    let total_area = (h * w) as u32;
    let min_area = (total_area / 10000).max(16);
    let max_area = total_area / 2;

    let mut components: Vec<BBox> = label_bboxes
        .values()
        .filter_map(|&(min_x, min_y, max_x, max_y)| {
            let bw = max_x - min_x + 1;
            let bh = max_y - min_y + 1;
            let area = bw * bh;
            if area < min_area || area > max_area {
                return None;
            }
            Some(BBox::new(min_x, min_y, bw, bh))
        })
        .collect();

    // Sort by y coordinate for grouping
    components.sort_by_key(|b| b.y);

    // Group into lines by vertical overlap
    let mut lines: Vec<(u32, u32, u32, u32)> = Vec::new(); // (min_x, min_y, max_x, max_y)

    for comp in &components {
        let cy_min = comp.y;
        let cy_max = comp.y + comp.height;
        let cx_min = comp.x;
        let cx_max = comp.x + comp.width;

        let mut merged = false;
        for line in &mut lines {
            let (_, ly_min, _, ly_max) = *line;
            let line_h = ly_max - ly_min;
            let comp_h = cy_max - cy_min;

            // Compute vertical overlap
            let overlap_start = cy_min.max(ly_min);
            let overlap_end = cy_max.min(ly_max);
            if overlap_end > overlap_start {
                let overlap = overlap_end - overlap_start;
                let min_h = line_h.min(comp_h);
                // Merge if overlap >= 50% of the smaller height
                if min_h > 0 && overlap * 2 >= min_h {
                    line.0 = line.0.min(cx_min);
                    line.1 = line.1.min(cy_min);
                    line.2 = line.2.max(cx_max);
                    line.3 = line.3.max(cy_max);
                    merged = true;
                    break;
                }
            }
        }

        if !merged {
            lines.push((cx_min, cy_min, cx_max, cy_max));
        }
    }

    // Convert to BBox and sort by y
    let mut result: Vec<BBox> = lines
        .into_iter()
        .map(|(min_x, min_y, max_x, max_y)| BBox::new(min_x, min_y, max_x - min_x, max_y - min_y))
        .collect();
    result.sort_by_key(|b| b.y);

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn empty_image_returns_empty() {
        let img = Array2::from_elem((0, 0), 255u8);
        assert!(detect_lines_ccl(&img).is_empty());
    }

    #[test]
    fn all_white_returns_empty() {
        let img = Array2::from_elem((100, 100), 255u8);
        assert!(detect_lines_ccl(&img).is_empty());
    }

    #[test]
    fn two_separate_blocks_detected_as_two_lines() {
        // 200x400 image with two text blocks at different y positions
        let mut img = Array2::from_elem((200, 400), 255u8);

        // Block 1: y=20..40, x=50..150 (several "characters")
        for y in 20..40 {
            for x in 50..70 {
                img[[y, x]] = 0;
            }
            for x in 80..100 {
                img[[y, x]] = 0;
            }
            for x in 110..130 {
                img[[y, x]] = 0;
            }
        }

        // Block 2: y=120..140, x=50..150
        for y in 120..140 {
            for x in 50..70 {
                img[[y, x]] = 0;
            }
            for x in 80..100 {
                img[[y, x]] = 0;
            }
        }

        let lines = detect_lines_ccl(&img);
        assert_eq!(lines.len(), 2, "Should detect 2 lines, got {:?}", lines);
        assert!(lines[0].y < lines[1].y, "Lines should be sorted by y");
    }

    #[test]
    fn single_line_multiple_chars_merged() {
        // Characters spread across x but same y range -> 1 line
        let mut img = Array2::from_elem((100, 400), 255u8);

        for y in 30..50 {
            for x in 10..30 {
                img[[y, x]] = 0;
            }
            for x in 200..220 {
                img[[y, x]] = 0;
            }
            for x in 350..370 {
                img[[y, x]] = 0;
            }
        }

        let lines = detect_lines_ccl(&img);
        assert_eq!(lines.len(), 1, "Should merge into 1 line, got {:?}", lines);
    }

    #[test]
    fn noise_filtered() {
        // Very small dots (1-2px) should be filtered out
        let mut img = Array2::from_elem((200, 200), 255u8);

        // Noise: single pixels
        img[[10, 10]] = 0;
        img[[50, 80]] = 0;
        img[[150, 150]] = 0;

        // 2x2 dot
        img[[100, 100]] = 0;
        img[[100, 101]] = 0;
        img[[101, 100]] = 0;
        img[[101, 101]] = 0;

        let lines = detect_lines_ccl(&img);
        assert!(
            lines.is_empty(),
            "Noise should be filtered, got {:?}",
            lines
        );
    }
}
