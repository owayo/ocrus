use ndarray::Array2;
use ocrus_core::BBox;
use ocrus_core::error::Result;

use crate::segment::segment_characters;

/// Result of cascade recognition for a single line.
pub struct CascadeResult {
    pub text: String,
    pub confidence: f32,
    pub used_ctc: bool,
}

/// Cascade recognizer: try fast classifier first, fallback to CTC.
pub struct CascadeRecognizer {
    threshold: f32,
}

impl CascadeRecognizer {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Get the confidence threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Recognize a line using cascade approach.
    ///
    /// Currently uses segment_characters for splitting,
    /// but the actual classifier inference is a placeholder
    /// until the cascade model is trained.
    pub fn recognize_line(
        &self,
        line_binary: &Array2<u8>,
        _line_bbox: &BBox,
    ) -> Result<CascadeResult> {
        let char_bboxes = segment_characters(line_binary);

        if char_bboxes.is_empty() {
            return Ok(CascadeResult {
                text: String::new(),
                confidence: 0.0,
                used_ctc: false,
            });
        }

        // Placeholder: return empty result indicating CTC should be used.
        // The actual classifier will be integrated when the cascade model
        // is trained (Phase 2.5).
        Ok(CascadeResult {
            text: String::new(),
            confidence: 0.0,
            used_ctc: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn cascade_empty_image() {
        let img = Array2::from_elem((48, 200), 255u8);
        let cascade = CascadeRecognizer::new(0.95);
        let result = cascade
            .recognize_line(&img, &BBox::new(0, 0, 200, 48))
            .unwrap();
        assert!(result.text.is_empty());
    }

    #[test]
    fn cascade_always_falls_back_to_ctc() {
        let mut img = Array2::from_elem((48, 200), 255u8);
        for y in 10..40 {
            for x in 20..50 {
                img[[y, x]] = 0;
            }
            for x in 70..100 {
                img[[y, x]] = 0;
            }
        }
        let cascade = CascadeRecognizer::new(0.95);
        let result = cascade
            .recognize_line(&img, &BBox::new(0, 0, 200, 48))
            .unwrap();
        assert!(result.used_ctc);
    }
}
