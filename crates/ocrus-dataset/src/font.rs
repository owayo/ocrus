use std::path::PathBuf;

use ab_glyph::{Font, FontRef};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Font style classification for training data diversity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FontStyle {
    /// 明朝体 / Serif
    Mincho,
    /// ゴシック体 / Sans-serif
    Gothic,
    /// 筆書体 / Script / Brush
    Script,
    /// モノスペース
    Monospace,
    /// その他 / 分類不明
    Other,
}

impl FontStyle {
    /// Classify a font by its file name (heuristic)
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        // Mincho / Serif patterns
        if lower.contains("mincho")
            || lower.contains("明朝")
            || lower.contains("serif")
            || lower.contains("song")
            || lower.contains("batang")
        {
            return Self::Mincho;
        }
        // Gothic / Sans-serif patterns
        if lower.contains("gothic")
            || lower.contains("ゴシック")
            || lower.contains("sans")
            || lower.contains("kaku")
            || lower.contains("maru")
            || lower.contains("hiraginosans")
            || lower.contains("yugothic")
        {
            return Self::Gothic;
        }
        // Script / Brush / Calligraphy patterns
        if lower.contains("script")
            || lower.contains("brush")
            || lower.contains("筆")
            || lower.contains("gyosho")
            || lower.contains("kaisho")
            || lower.contains("cursive")
            || lower.contains("handwrit")
        {
            return Self::Script;
        }
        // Monospace patterns
        if lower.contains("mono")
            || lower.contains("courier")
            || lower.contains("consolas")
            || lower.contains("menlo")
            || lower.contains("source code")
        {
            return Self::Monospace;
        }
        Self::Other
    }

    /// Return all styles
    pub fn all() -> &'static [FontStyle] {
        &[
            Self::Mincho,
            Self::Gothic,
            Self::Script,
            Self::Monospace,
            Self::Other,
        ]
    }
}

impl std::fmt::Display for FontStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mincho => write!(f, "mincho"),
            Self::Gothic => write!(f, "gothic"),
            Self::Script => write!(f, "script"),
            Self::Monospace => write!(f, "monospace"),
            Self::Other => write!(f, "other"),
        }
    }
}

pub struct FontEntry {
    pub name: String,
    pub path: PathBuf,
    pub data: Vec<u8>,
    pub index: u32,
    pub style: FontStyle,
}

impl FontEntry {
    pub fn font_ref(&self) -> Result<FontRef<'_>> {
        Ok(FontRef::try_from_slice(&self.data)?)
    }
}

pub fn default_font_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![
        PathBuf::from("/System/Library/Fonts"),
        PathBuf::from("/Library/Fonts"),
    ];
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join("Library/Fonts"));
    }
    dirs
}

/// Discover fonts, optionally filtered by style
pub fn discover_fonts_filtered(dirs: &[PathBuf], styles: Option<&[FontStyle]>) -> Vec<FontEntry> {
    let mut fonts = discover_fonts(dirs);
    if let Some(styles) = styles {
        fonts.retain(|f| styles.contains(&f.style));
    }
    fonts
}

pub fn discover_fonts(dirs: &[PathBuf]) -> Vec<FontEntry> {
    let mut entries = Vec::new();
    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        let Ok(read_dir) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in read_dir.flatten() {
            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase());
            match ext.as_deref() {
                Some("ttf" | "otf" | "ttc") => {}
                _ => continue,
            }
            let Ok(data) = std::fs::read(&path) else {
                continue;
            };
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            if let Some(fe) = try_load_font(name, path, data, 0) {
                entries.push(fe);
            }
        }
    }
    entries
}

fn try_load_font(name: String, path: PathBuf, data: Vec<u8>, index: u32) -> Option<FontEntry> {
    let font_ref = FontRef::try_from_slice(&data).ok()?;
    // Check if font supports Japanese (hiragana 'あ' U+3042)
    let glyph_id = font_ref.glyph_id('あ');
    if glyph_id.0 == 0 {
        return None;
    }
    let style = FontStyle::from_name(&name);
    Some(FontEntry {
        name,
        path,
        data,
        index,
        style,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_fonts_empty_dir() {
        let fonts = discover_fonts(&[PathBuf::from("/nonexistent_dir_12345")]);
        assert!(fonts.is_empty());
    }

    #[test]
    fn default_dirs_not_empty() {
        let dirs = default_font_dirs();
        assert!(!dirs.is_empty());
    }
}
