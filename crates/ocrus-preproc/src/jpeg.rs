/// JPEG detection utility.
/// Provides fast JPEG format detection by magic bytes.
use image::DynamicImage;

/// Check if data is JPEG format and decode using the image crate.
/// Returns `None` if not JPEG or decoding fails.
pub fn try_decode_jpeg(data: &[u8]) -> Option<DynamicImage> {
    // Check JPEG magic bytes (FFD8FF)
    if data.len() < 3 || data[0] != 0xFF || data[1] != 0xD8 || data[2] != 0xFF {
        return None;
    }

    image::load_from_memory_with_format(data, image::ImageFormat::Jpeg).ok()
}

/// Check if data starts with JPEG magic bytes.
pub fn is_jpeg(data: &[u8]) -> bool {
    data.len() >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_jpeg_returns_none() {
        let png_data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert!(try_decode_jpeg(&png_data).is_none());
    }

    #[test]
    fn test_empty_returns_none() {
        assert!(try_decode_jpeg(&[]).is_none());
    }

    #[test]
    fn test_short_data_returns_none() {
        assert!(try_decode_jpeg(&[0xFF, 0xD8]).is_none());
    }

    #[test]
    fn test_is_jpeg() {
        assert!(is_jpeg(&[0xFF, 0xD8, 0xFF, 0xE0]));
        assert!(!is_jpeg(&[0x89, 0x50, 0x4E, 0x47]));
        assert!(!is_jpeg(&[]));
    }

    #[test]
    fn test_invalid_jpeg_returns_none() {
        let data = [0xFF, 0xD8, 0xFF, 0x00, 0x00];
        assert!(try_decode_jpeg(&data).is_none());
    }
}
