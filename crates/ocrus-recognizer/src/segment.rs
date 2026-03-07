use ndarray::Array2;
use ocrus_core::BBox;

/// Segment a binary line image into individual character bounding boxes
/// using vertical projection profile analysis.
///
/// Input: binary image where 0 = foreground (text), 255 = background.
/// Returns bounding boxes for each detected character region.
pub fn segment_characters(line_binary: &Array2<u8>) -> Vec<BBox> {
    let (h, w) = line_binary.dim();
    if h == 0 || w == 0 {
        return Vec::new();
    }

    // Vertical projection: count foreground pixels per column
    let projection: Vec<usize> = (0..w)
        .map(|c| line_binary.column(c).iter().filter(|&&px| px == 0).count())
        .collect();

    // Compute median of non-zero projection values for threshold
    let mut nonzero: Vec<usize> = projection.iter().copied().filter(|&v| v > 0).collect();
    if nonzero.is_empty() {
        return Vec::new();
    }
    nonzero.sort_unstable();
    let median = nonzero[nonzero.len() / 2];
    let gap_threshold = median / 10; // 10% of median

    // Find character regions (runs of columns above the gap threshold)
    let min_width = (h as f64 * 0.2) as usize;
    let mut regions = Vec::new();
    let mut start: Option<usize> = None;

    for (x, &count) in projection.iter().enumerate() {
        if count > gap_threshold {
            if start.is_none() {
                start = Some(x);
            }
        } else if let Some(s) = start.take() {
            let region_w = x - s;
            if region_w >= min_width {
                regions.push(BBox::new(s as u32, 0, region_w as u32, h as u32));
            }
        }
    }

    // Handle trailing region
    if let Some(s) = start {
        let region_w = w - s;
        if region_w >= min_width {
            regions.push(BBox::new(s as u32, 0, region_w as u32, h as u32));
        }
    }

    regions
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn empty_image_returns_empty() {
        let img = Array2::from_elem((0, 0), 255u8);
        assert!(segment_characters(&img).is_empty());
    }

    #[test]
    fn all_white_returns_empty() {
        let img = Array2::from_elem((30, 100), 255u8);
        assert!(segment_characters(&img).is_empty());
    }

    #[test]
    fn three_characters_segmented() {
        let mut img = Array2::from_elem((30, 100), 255u8);

        // Char 1: x=5..20
        for y in 5..25 {
            for x in 5..20 {
                img[[y, x]] = 0;
            }
        }
        // Char 2: x=35..50
        for y in 5..25 {
            for x in 35..50 {
                img[[y, x]] = 0;
            }
        }
        // Char 3: x=65..80
        for y in 5..25 {
            for x in 65..80 {
                img[[y, x]] = 0;
            }
        }

        let bboxes = segment_characters(&img);
        assert_eq!(
            bboxes.len(),
            3,
            "Should detect 3 characters, got {:?}",
            bboxes
        );
        assert!(bboxes[0].x < bboxes[1].x);
        assert!(bboxes[1].x < bboxes[2].x);
    }

    #[test]
    fn noise_filtered_by_min_width() {
        let mut img = Array2::from_elem((50, 100), 255u8);

        // Narrow noise: width=2 (below 50*0.2=10 min width)
        for y in 10..40 {
            for x in 20..22 {
                img[[y, x]] = 0;
            }
        }

        // Valid character: width=15
        for y in 10..40 {
            for x in 50..65 {
                img[[y, x]] = 0;
            }
        }

        let bboxes = segment_characters(&img);
        assert_eq!(
            bboxes.len(),
            1,
            "Narrow noise should be filtered, got {:?}",
            bboxes
        );
    }
}
