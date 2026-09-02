//! Morphological filters on grayscale images.
//!
//! Text is dark on light here, so a 3x3 minimum filter thickens strokes and a maximum
//! filter thins them. Recognizing the same crop under both gives the model two more views
//! of a glyph whose strokes are too thin or too heavy for the size it was rendered at.

use ndarray::Array2;

/// Replace each pixel with the smallest value in its 3x3 neighbourhood (thicker strokes).
pub fn thicken(src: &Array2<u8>) -> Array2<u8> {
    filter_3x3(src, u8::min, u8::MAX)
}

/// Replace each pixel with the largest value in its 3x3 neighbourhood (thinner strokes).
pub fn thin(src: &Array2<u8>) -> Array2<u8> {
    filter_3x3(src, u8::max, u8::MIN)
}

fn filter_3x3(src: &Array2<u8>, pick: fn(u8, u8) -> u8, init: u8) -> Array2<u8> {
    let (h, w) = src.dim();
    let mut out = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(h - 1);
        for x in 0..w {
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(w - 1);
            let mut acc = init;
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    acc = pick(acc, src[[yy, xx]]);
                }
            }
            out[[y, x]] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thicken_spreads_dark_pixels() {
        let mut src = Array2::<u8>::from_elem((3, 3), 255);
        src[[1, 1]] = 0;
        let out = thicken(&src);
        assert!(
            out.iter().all(|&v| v == 0),
            "a dark pixel should fill a 3x3 neighbourhood"
        );
    }

    #[test]
    fn thin_removes_isolated_dark_pixels() {
        let mut src = Array2::<u8>::from_elem((3, 3), 255);
        src[[1, 1]] = 0;
        let out = thin(&src);
        assert!(
            out.iter().all(|&v| v == 255),
            "an isolated dark pixel should disappear"
        );
    }
}
