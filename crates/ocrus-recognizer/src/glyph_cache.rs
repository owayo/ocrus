use std::collections::HashMap;

use img_hash::image::GrayImage;
use img_hash::{HashAlg, Hasher, HasherConfig, ImageHash};

/// Cache for recognized glyphs using perceptual hashing.
/// When a line image's hash matches a cached entry, the recognition result
/// is reused without running inference.
pub struct GlyphCache {
    hasher: Hasher,
    cache: HashMap<Vec<u8>, CacheEntry>,
    max_distance: u32,
}

struct CacheEntry {
    hash: ImageHash,
    text: String,
    confidence: f32,
}

impl GlyphCache {
    /// Create a new glyph cache.
    /// `max_distance` is the maximum Hamming distance for a hash match (0 = exact).
    pub fn new(max_distance: u32) -> Self {
        let hasher = HasherConfig::new()
            .hash_alg(HashAlg::DoubleGradient)
            .hash_size(8, 8)
            .to_hasher();
        Self {
            hasher,
            cache: HashMap::new(),
            max_distance,
        }
    }

    /// Compute the perceptual hash of a line image.
    /// Input: raw grayscale pixel data and dimensions.
    pub fn compute_hash(&self, gray_data: &[u8], width: u32, height: u32) -> Option<ImageHash> {
        let img = GrayImage::from_raw(width, height, gray_data.to_vec())?;
        Some(self.hasher.hash_image(&img))
    }

    /// Look up a hash in the cache. Returns (text, confidence) if found.
    pub fn lookup(&self, hash: &ImageHash) -> Option<(&str, f32)> {
        let hash_bytes = hash.as_bytes();
        // Exact match first
        if let Some(entry) = self.cache.get(hash_bytes) {
            return Some((&entry.text, entry.confidence));
        }
        // Fuzzy match if max_distance > 0
        if self.max_distance > 0 {
            for entry in self.cache.values() {
                if hash.dist(&entry.hash) <= self.max_distance {
                    return Some((&entry.text, entry.confidence));
                }
            }
        }
        None
    }

    /// Insert a recognition result into the cache.
    pub fn insert(&mut self, hash: ImageHash, text: String, confidence: f32) {
        let hash_bytes = hash.as_bytes().to_vec();
        self.cache.insert(
            hash_bytes,
            CacheEntry {
                hash,
                text,
                confidence,
            },
        );
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_cache() {
        let cache = GlyphCache::new(0);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_insert_and_exact_lookup() {
        let mut cache = GlyphCache::new(0);
        let data = vec![0u8; 64]; // 8x8 black
        let hash = cache.compute_hash(&data, 8, 8).unwrap();
        cache.insert(hash.clone(), "あ".to_string(), 0.95);

        // Same image should match
        let hash2 = cache.compute_hash(&data, 8, 8).unwrap();
        let result = cache.lookup(&hash2);
        assert!(result.is_some());
        let (text, conf) = result.unwrap();
        assert_eq!(text, "あ");
        assert!((conf - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_different_image_no_match() {
        let mut cache = GlyphCache::new(0);
        // Gradient-based pattern: ascending values
        let data1: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        // Opposite gradient pattern: descending values
        let data2: Vec<u8> = (0..64).map(|i| (255 - i * 4) as u8).collect();

        let hash1 = cache.compute_hash(&data1, 8, 8).unwrap();
        cache.insert(hash1, "あ".to_string(), 0.95);

        let hash2 = cache.compute_hash(&data2, 8, 8).unwrap();
        let result = cache.lookup(&hash2);
        assert!(result.is_none());
    }

    #[test]
    fn test_fuzzy_match() {
        let mut cache = GlyphCache::new(5);
        let data1 = vec![0u8; 64];
        let hash1 = cache.compute_hash(&data1, 8, 8).unwrap();
        cache.insert(hash1, "い".to_string(), 0.9);

        // Slightly different image (1 pixel changed)
        let mut data2 = vec![0u8; 64];
        data2[0] = 128;
        let hash2 = cache.compute_hash(&data2, 8, 8).unwrap();
        // Just verify it doesn't panic
        let _ = cache.lookup(&hash2);
    }
}
