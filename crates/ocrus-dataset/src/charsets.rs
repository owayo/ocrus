use std::path::Path;

use anyhow::{Context, Result};

pub fn load_charset(data_dir: &Path, category: &str) -> Result<Vec<char>> {
    let path = data_dir.join(format!("{category}.txt"));
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read charset file: {}", path.display()))?;
    let chars: Vec<char> = text.trim().chars().collect();
    Ok(chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_charset_missing_file() {
        let result = load_charset(Path::new("/nonexistent"), "missing");
        assert!(result.is_err());
    }

    #[test]
    fn load_charset_from_data_dir() {
        let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/test_chars");
        if data_dir.exists() {
            let chars = load_charset(&data_dir, "hiragana").unwrap();
            assert!(!chars.is_empty());
            assert!(chars.contains(&'あ'));
        }
    }
}
