use std::path::PathBuf;

use ab_glyph::{Font, FontRef};
use anyhow::Result;

pub struct FontEntry {
    pub name: String,
    pub path: PathBuf,
    pub data: Vec<u8>,
    pub index: u32,
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
    Some(FontEntry {
        name,
        path,
        data,
        index,
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
