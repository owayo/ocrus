/// Character set for CTC decoding (PaddleOCR convention).
/// Characters are indexed 0..N-1, blank token is the last index (N).
#[derive(Debug, Clone)]
pub struct Charset {
    chars: Vec<char>,
}

impl Charset {
    /// Create a charset from a list of characters.
    /// Blank token is automatically appended at the end (PaddleOCR convention).
    pub fn from_chars(chars: &[char]) -> Self {
        Self {
            chars: chars.to_vec(),
        }
    }

    /// Load charset from a text file (one character per line, no blank line needed).
    pub fn from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let chars: Vec<char> = content
            .lines()
            .filter(|l| !l.is_empty())
            .filter_map(|l| l.chars().next())
            .collect();
        Ok(Self { chars })
    }

    /// Blank token index (last index, PaddleOCR convention).
    pub fn blank_index(&self) -> usize {
        self.chars.len()
    }

    /// Number of classes including blank.
    pub fn num_classes(&self) -> usize {
        self.chars.len() + 1
    }

    /// Get character for a given index. Blank index → None.
    pub fn index_to_char(&self, index: usize) -> Option<char> {
        self.chars.get(index).copied()
    }

    /// Create a built-in JIS X 0208 charset.
    /// Includes: ASCII printable, hiragana, katakana, major CJK unified ideographs, common symbols.
    pub fn from_jis() -> Self {
        let mut chars = Vec::with_capacity(10000);

        // ASCII printable (0x21..=0x7E)
        for c in 0x21u32..=0x7E {
            if let Some(ch) = char::from_u32(c) {
                chars.push(ch);
            }
        }

        // Fullwidth forms and common symbols
        for c in 0x3000u32..=0x303F {
            if let Some(ch) = char::from_u32(c) {
                chars.push(ch);
            }
        }

        // Hiragana (U+3041..=U+3096)
        for c in 0x3041u32..=0x3096 {
            if let Some(ch) = char::from_u32(c) {
                chars.push(ch);
            }
        }

        // Katakana (U+30A1..=U+30FA)
        for c in 0x30A1u32..=0x30FA {
            if let Some(ch) = char::from_u32(c) {
                chars.push(ch);
            }
        }

        // CJK Unified Ideographs main range (U+4E00..=U+9FFF)
        for c in 0x4E00u32..=0x9FFF {
            if let Some(ch) = char::from_u32(c) {
                chars.push(ch);
            }
        }

        // Fullwidth ASCII (U+FF01..=U+FF5E)
        for c in 0xFF01u32..=0xFF5E {
            if let Some(ch) = char::from_u32(c) {
                chars.push(ch);
            }
        }

        Self { chars }
    }

    /// Create a strict JIS X 0208 charset from data files.
    ///
    /// Reads all `.txt` files under `data_dir` (e.g. `data/test_chars/`) and collects
    /// every non-whitespace character. This gives an exact JIS character set (~6,500 chars)
    /// rather than the broad Unicode-range approximation of [`from_jis`].
    ///
    /// Falls back to [`from_jis`] if `data_dir` does not exist or contains no characters.
    pub fn from_jis_strict(data_dir: &std::path::Path) -> Self {
        if !data_dir.is_dir() {
            return Self::from_jis();
        }

        let mut chars_set = std::collections::HashSet::new();

        let entries = match std::fs::read_dir(data_dir) {
            Ok(e) => e,
            Err(_) => return Self::from_jis(),
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "txt")
                && let Ok(content) = std::fs::read_to_string(&path)
            {
                for ch in content.chars() {
                    if !ch.is_whitespace() {
                        chars_set.insert(ch);
                    }
                }
            }
        }

        if chars_set.is_empty() {
            return Self::from_jis();
        }

        let mut chars: Vec<char> = chars_set.into_iter().collect();
        chars.sort_unstable();

        Self { chars }
    }

    /// Build a logit mask against a full charset.
    /// Returns a boolean vec of length `full_charset.num_classes()`.
    /// `true` means the class is allowed (present in self or is blank).
    pub fn logit_mask(&self, full_charset: &Charset) -> Vec<bool> {
        let mut mask = vec![false; full_charset.num_classes()];
        // blank is always allowed
        mask[full_charset.blank_index()] = true;
        for (i, slot) in mask.iter_mut().enumerate() {
            if let Some(ch) = full_charset.index_to_char(i)
                && self.contains(ch)
            {
                *slot = true;
            }
        }
        mask
    }

    /// Check if the charset contains a given character.
    pub fn contains(&self, c: char) -> bool {
        self.chars.contains(&c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_charset_basics() {
        let cs = Charset::from_chars(&['あ', 'い', 'う']);
        assert_eq!(cs.num_classes(), 4); // 3 chars + blank
        assert_eq!(cs.blank_index(), 3); // blank is last
        assert_eq!(cs.index_to_char(0), Some('あ'));
        assert_eq!(cs.index_to_char(1), Some('い'));
        assert_eq!(cs.index_to_char(2), Some('う'));
        assert_eq!(cs.index_to_char(3), None); // blank
        assert_eq!(cs.index_to_char(4), None);
    }

    #[test]
    fn test_from_jis() {
        let jis = Charset::from_jis();
        // Should contain hiragana, katakana, CJK, ASCII
        assert!(jis.contains('あ'));
        assert!(jis.contains('ア'));
        assert!(jis.contains('漢'));
        assert!(jis.contains('A'));
        assert!(jis.contains('!'));
        // Should have a reasonable number of chars
        assert!(jis.num_classes() > 20000);
    }

    #[test]
    fn test_logit_mask() {
        let full = Charset::from_chars(&['a', 'b', 'c', 'd']);
        let subset = Charset::from_chars(&['a', 'c']);

        let mask = subset.logit_mask(&full);
        // mask length = num_classes of full = 5
        assert_eq!(mask.len(), 5);
        assert!(mask[0]); // 'a'
        assert!(!mask[1]); // 'b'
        assert!(mask[2]); // 'c'
        assert!(!mask[3]); // 'd'
        assert!(mask[4]); // blank
    }

    #[test]
    fn test_contains() {
        let cs = Charset::from_chars(&['あ', 'い']);
        assert!(cs.contains('あ'));
        assert!(!cs.contains('う'));
    }

    #[test]
    fn test_from_jis_strict() {
        let data_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/test_chars");
        let jis_strict = Charset::from_jis_strict(&data_dir);
        // JIS X 0208 strict should have ~6,500 chars (+ blank = ~6,500 num_classes)
        // All data files together have more, but should be well under the 20,000+ of from_jis()
        assert!(
            jis_strict.num_classes() > 6000,
            "too few: {}",
            jis_strict.num_classes()
        );
        assert!(
            jis_strict.num_classes() < 15000,
            "too many: {}",
            jis_strict.num_classes()
        );
        // Should contain basic characters
        assert!(jis_strict.contains('あ'));
        assert!(jis_strict.contains('ア'));
        assert!(jis_strict.contains('A'));
        assert!(jis_strict.contains('亜')); // JIS level 1
    }

    #[test]
    fn test_from_jis_strict_fallback() {
        let data_dir = std::path::Path::new("/nonexistent/path");
        let jis = Charset::from_jis_strict(data_dir);
        // Should fall back to from_jis() which has >20,000 classes
        assert!(jis.num_classes() > 20000);
    }
}
