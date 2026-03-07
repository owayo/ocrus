use image::{GrayImage, Luma};
use imageproc::region_labelling::{Connectivity, connected_components};
use ndarray::Array2;
use ocrus_core::BBox;

use crate::vertical::TextOrientation;

pub struct RubySeparation {
    pub body_bbox: BBox,
    pub ruby_bboxes: Vec<BBox>,
}

/// Separate ruby (furigana) annotations from body text within a line.
///
/// Detects small connected components positioned above (horizontal) or
/// to the right (vertical) of the main text line and classifies them as ruby.
pub fn separate_ruby(
    binary: &Array2<u8>,
    line_bbox: &BBox,
    orientation: TextOrientation,
) -> RubySeparation {
    let (img_h, img_w) = binary.dim();

    // Clamp line_bbox to image bounds
    let x0 = (line_bbox.x as usize).min(img_w);
    let y0 = (line_bbox.y as usize).min(img_h);
    let x1 = ((line_bbox.x + line_bbox.width) as usize).min(img_w);
    let y1 = ((line_bbox.y + line_bbox.height) as usize).min(img_h);

    let crop_w = x1 - x0;
    let crop_h = y1 - y0;

    if crop_w == 0 || crop_h == 0 {
        return RubySeparation {
            body_bbox: *line_bbox,
            ruby_bboxes: vec![],
        };
    }

    // Extract cropped region as GrayImage
    let crop_img = GrayImage::from_fn(crop_w as u32, crop_h as u32, |x, y| {
        Luma([binary[[y0 + y as usize, x0 + x as usize]]])
    });

    // Run CCL
    let labeled = connected_components(&crop_img, Connectivity::Eight, Luma([255u8]));

    // Compute bounding box for each component
    let mut component_bboxes: std::collections::HashMap<u32, (u32, u32, u32, u32)> =
        std::collections::HashMap::new();

    for y in 0..crop_h as u32 {
        for x in 0..crop_w as u32 {
            let label = labeled.get_pixel(x, y).0[0];
            if label == 0 {
                continue;
            }
            component_bboxes
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

    if component_bboxes.is_empty() {
        return RubySeparation {
            body_bbox: *line_bbox,
            ruby_bboxes: vec![],
        };
    }

    // Compute heights of all components
    let mut heights: Vec<u32> = component_bboxes
        .values()
        .map(|&(_, min_y, _, max_y)| max_y - min_y + 1)
        .collect();
    heights.sort_unstable();

    let median_height = heights[heights.len() / 2];

    // Threshold: components with height <= 65% of median are ruby candidates
    let ruby_threshold = (median_height as f64 * 0.65) as u32;

    // Line center for position check
    let center_y = crop_h as u32 / 2;
    let center_x = crop_w as u32 / 2;

    let mut ruby_bboxes = Vec::new();
    let mut body_min_x = u32::MAX;
    let mut body_min_y = u32::MAX;
    let mut body_max_x = 0u32;
    let mut body_max_y = 0u32;
    let mut has_body = false;

    for &(min_x, min_y, max_x, max_y) in component_bboxes.values() {
        let comp_h = max_y - min_y + 1;
        let comp_center_y = (min_y + max_y) / 2;
        let comp_center_x = (min_x + max_x) / 2;

        let is_ruby_size = comp_h <= ruby_threshold;
        let is_ruby_position = match orientation {
            TextOrientation::Vertical => comp_center_x > center_x, // right side
            _ => comp_center_y < center_y,                         // above center
        };

        if is_ruby_size && is_ruby_position {
            // Convert to original image coordinates
            ruby_bboxes.push(BBox::new(
                min_x + x0 as u32,
                min_y + y0 as u32,
                max_x - min_x + 1,
                max_y - min_y + 1,
            ));
        } else {
            has_body = true;
            body_min_x = body_min_x.min(min_x);
            body_min_y = body_min_y.min(min_y);
            body_max_x = body_max_x.max(max_x);
            body_max_y = body_max_y.max(max_y);
        }
    }

    let body_bbox = if has_body {
        BBox::new(
            body_min_x + x0 as u32,
            body_min_y + y0 as u32,
            body_max_x - body_min_x + 1,
            body_max_y - body_min_y + 1,
        )
    } else {
        *line_bbox
    };

    RubySeparation {
        body_bbox,
        ruby_bboxes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn ruby_detected_horizontal() {
        // 200x50 image: small components at top (ruby), large at bottom (body)
        let mut img = Array2::from_elem((50, 200), 255u8);

        // Ruby: small components at y=5..15 (height=10)
        for y in 5..15 {
            for x in 20..30 {
                img[[y, x]] = 0;
            }
            for x in 40..50 {
                img[[y, x]] = 0;
            }
        }

        // Body: large components at y=20..45 (height=25)
        for y in 20..45 {
            for x in 10..40 {
                img[[y, x]] = 0;
            }
            for x in 50..80 {
                img[[y, x]] = 0;
            }
        }

        let line_bbox = BBox::new(0, 0, 200, 50);
        let result = separate_ruby(&img, &line_bbox, TextOrientation::Horizontal);

        assert!(
            !result.ruby_bboxes.is_empty(),
            "Should detect ruby components, got none"
        );
    }

    #[test]
    fn no_ruby_when_uniform_size() {
        // All components same size -> no ruby
        let mut img = Array2::from_elem((50, 200), 255u8);

        for y in 15..35 {
            for x in 10..30 {
                img[[y, x]] = 0;
            }
            for x in 50..70 {
                img[[y, x]] = 0;
            }
            for x in 90..110 {
                img[[y, x]] = 0;
            }
        }

        let line_bbox = BBox::new(0, 0, 200, 50);
        let result = separate_ruby(&img, &line_bbox, TextOrientation::Horizontal);

        assert!(
            result.ruby_bboxes.is_empty(),
            "Should not detect ruby in uniform-size components, got {:?}",
            result.ruby_bboxes.len()
        );
    }

    #[test]
    fn ruby_detected_vertical() {
        // Vertical text: ruby on the right side
        let mut img = Array2::from_elem((200, 80), 255u8);

        // Body: large components on the left (x=5..35, height=25 each)
        for y in 10..35 {
            for x in 5..35 {
                img[[y, x]] = 0;
            }
        }
        for y in 50..75 {
            for x in 5..35 {
                img[[y, x]] = 0;
            }
        }

        // Ruby: small components on the right (x=50..60, height=10 each)
        for y in 15..25 {
            for x in 50..60 {
                img[[y, x]] = 0;
            }
        }
        for y in 55..65 {
            for x in 50..60 {
                img[[y, x]] = 0;
            }
        }

        let line_bbox = BBox::new(0, 0, 80, 200);
        let result = separate_ruby(&img, &line_bbox, TextOrientation::Vertical);

        assert!(
            !result.ruby_bboxes.is_empty(),
            "Should detect ruby in vertical text, got none"
        );
    }
}
