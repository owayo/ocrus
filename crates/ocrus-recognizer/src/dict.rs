use daachorse::{CharwiseDoubleArrayAhoCorasick, CharwiseDoubleArrayAhoCorasickBuilder, MatchKind};
use std::io;
use std::path::Path;

/// Aho-Corasick 法で複数の誤認識パターンを検索し、OCR 出力を補正する。
pub struct DictCorrector {
    automaton: Option<CharwiseDoubleArrayAhoCorasick<usize>>,
    replacements: Vec<String>,
}

impl DictCorrector {
    /// 各行が `誤認識文字列\t正しい文字列` の補正ルールを読み込む。
    /// 空行とタブのない行は読み飛ばす。
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

    /// 左端の最長一致から順に、重複しない範囲を補正する。
    /// 置換後の文字列に補正ルールを再適用しない。
    pub fn correct(&self, text: &str) -> String {
        let automaton = match &self.automaton {
            Some(a) => a,
            None => return text.to_string(),
        };

        let mut result = String::with_capacity(text.len());
        let mut pos = 0;

        // 一致位置は UTF-8 の文字境界を指すバイト位置なので、そのまま切り出せる。
        for m in automaton.leftmost_find_iter(text) {
            result.push_str(&text[pos..m.start()]);
            result.push_str(&self.replacements[m.value()]);
            pos = m.end();
        }

        result.push_str(&text[pos..]);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn corrector(rules: &str) -> DictCorrector {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(rules.as_bytes()).unwrap();
        DictCorrector::from_file(file.path()).unwrap()
    }

    #[test]
    fn test_dict_corrector() {
        let corrector = corrector("こんにちわ\tこんにちは\n仕柿\t仕事\n");
        assert_eq!(corrector.correct("こんにちわ"), "こんにちは");
        assert_eq!(corrector.correct("仕柿中"), "仕事中");
        assert_eq!(corrector.correct("テスト"), "テスト");
    }

    #[test]
    fn test_dict_corrector_empty() {
        let corrector = corrector("");
        assert_eq!(corrector.correct("テスト"), "テスト");
        assert_eq!(corrector.correct(""), "");
    }

    #[test]
    fn corrections_preserve_utf8_and_do_not_reprocess_replacements() {
        let corrector = corrector("誤字\t正\nx\t誤字\n誤\t短\n");
        assert_eq!(
            corrector.correct("先頭🙂誤字x誤字🦀後尾"),
            "先頭🙂正誤字正🦀後尾"
        );
        assert_eq!(corrector.correct("誤字誤字"), "正正");
        assert_eq!(corrector.correct("e\u{301}🙂"), "e\u{301}🙂");
        assert_eq!(corrector.correct(""), "");
    }

    #[test]
    fn corrections_choose_the_longest_match_at_the_leftmost_position() {
        let corrector = corrector("あ\t短\nあい\t長\nいう\t別\n");
        assert_eq!(corrector.correct("あいう"), "長う");
        assert_eq!(corrector.correct("いうあ"), "別短");
    }

    #[test]
    fn corrections_handle_many_matches_and_keep_unmatched_text() {
        let corrector = corrector("誤字\t正しい語\n");
        assert_eq!(
            corrector.correct(&"🙂誤字/".repeat(1024)),
            "🙂正しい語/".repeat(1024)
        );
    }
}
