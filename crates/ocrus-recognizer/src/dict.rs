use daachorse::{CharwiseDoubleArrayAhoCorasick, CharwiseDoubleArrayAhoCorasickBuilder, MatchKind};
use std::io;
use std::path::Path;

/// Dictionary-based post-correction for OCR output.
/// Uses Aho-Corasick automaton for efficient multi-pattern matching.
pub struct DictCorrector {
    automaton: Option<CharwiseDoubleArrayAhoCorasick<usize>>,
    replacements: Vec<String>,
}

impl DictCorrector {
    /// Load correction rules from a file.
    /// File format: each line is `wrong_text\tcorrect_text`.
    /// Empty lines and lines without a tab are skipped.
    pub fn from_file(path: &Path) -> io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut patterns = Vec::new();
        let mut replacements = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some((wrong, correct)) = line.split_once('\t')
                && !wrong.is_empty()
            {
                patterns.push(wrong.to_string());
                replacements.push(correct.to_string());
            }
        }

        let automaton = if patterns.is_empty() {
            None
        } else {
            let patvals: Vec<(&str, usize)> = patterns
                .iter()
                .enumerate()
                .map(|(i, p)| (p.as_str(), i))
                .collect();
            let aut = CharwiseDoubleArrayAhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build_with_values(patvals)
                .map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("failed to build automaton: {e}"),
                    )
                })?;
            Some(aut)
        };

        Ok(Self {
            automaton,
            replacements,
        })
    }

    /// Apply corrections to text.
    /// Finds all occurrences of wrong patterns and replaces them with correct text.
    /// Processes left-to-right, non-overlapping (leftmost-longest).
    pub fn correct(&self, text: &str) -> String {
        let automaton = match &self.automaton {
            Some(a) => a,
            None => return text.to_string(),
        };

        let chars: Vec<char> = text.chars().collect();
        let mut result = String::with_capacity(text.len());
        let mut pos = 0;

        // daachorse returns byte positions; convert to char positions
        for m in automaton.leftmost_find_iter(text) {
            let cs = byte_to_char_pos(text, m.start());
            let ce = byte_to_char_pos(text, m.end());
            let val = m.value();

            if cs < pos {
                continue;
            }
            for &ch in &chars[pos..cs] {
                result.push(ch);
            }
            result.push_str(&self.replacements[val]);
            pos = ce;
        }

        for &ch in &chars[pos..] {
            result.push(ch);
        }

        result
    }
}

fn byte_to_char_pos(text: &str, byte_pos: usize) -> usize {
    text[..byte_pos].chars().count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_dict_corrector() {
        let dir = std::env::temp_dir().join("ocrus_dict_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corrections.txt");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "こんにちわ\tこんにちは").unwrap();
            writeln!(f, "仕柿\t仕事").unwrap();
        }

        let corrector = DictCorrector::from_file(&path).unwrap();
        assert_eq!(corrector.correct("こんにちわ"), "こんにちは");
        assert_eq!(corrector.correct("仕柿中"), "仕事中");
        assert_eq!(corrector.correct("テスト"), "テスト");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_dict_corrector_empty() {
        let dir = std::env::temp_dir().join("ocrus_dict_empty_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.txt");
        std::fs::File::create(&path).unwrap();

        let corrector = DictCorrector::from_file(&path).unwrap();
        assert_eq!(corrector.correct("テスト"), "テスト");
        std::fs::remove_dir_all(&dir).ok();
    }
}
