use ab_glyph::{FontRef, PxScale};
use image::{GrayImage, Luma};
use imageproc::drawing::{draw_text_mut, text_size};

const DEFAULT_HEIGHT: u32 = 48;
const PADDING: u32 = 4;

pub fn render_text_line(font: &FontRef<'_>, text: &str, height: u32) -> GrayImage {
    let scale = PxScale::from(height as f32 * 0.85);
    let (text_w, _text_h) = text_size(scale, font, text);
    let img_width = text_w + PADDING * 2;
    let img_height = height;
    let mut img = GrayImage::from_pixel(img_width, img_height, Luma([255u8]));
    let x = PADDING as i32;
    let y = (height as f32 * 0.075) as i32;
    draw_text_mut(&mut img, Luma([0u8]), x, y, scale, font, text);
    img
}

pub fn render_single_char(font: &FontRef<'_>, ch: char) -> GrayImage {
    render_text_line(font, &ch.to_string(), DEFAULT_HEIGHT)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_font() -> Option<(Vec<u8>, FontRef<'static>)> {
        // Try to load a system font for testing
        let paths = [
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ];
        for path in &paths {
            if let Ok(data) = std::fs::read(path)
                && let Ok(font) = FontRef::try_from_slice(Box::leak(data.into_boxed_slice()))
            {
                return Some((Vec::new(), font));
            }
        }
        None
    }

    #[test]
    fn render_produces_correct_height() {
        if let Some((_data, font)) = test_font() {
            let img = render_text_line(&font, "あ", 48);
            assert_eq!(img.height(), 48);
            assert!(img.width() > 0);
        }
    }
}
