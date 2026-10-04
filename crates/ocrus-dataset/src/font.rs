use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

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
        // Specific styles must precede generic ones: "sans-serif" contains
        // "serif", and "Noto Sans Mono" contains "sans".
        if lower.contains("mono")
            || lower.contains("courier")
            || lower.contains("consolas")
            || lower.contains("menlo")
            || lower.contains("source code")
        {
            return Self::Monospace;
        }
        if lower.contains("sans") {
            return Self::Gothic;
        }
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
    /// Shared across all faces of a font collection.
    pub data: Arc<[u8]>,
    pub index: u32,
    pub style: FontStyle,
}

impl FontEntry {
    pub fn font_ref(&self) -> Result<FontRef<'_>> {
        Ok(FontRef::try_from_slice_and_index(&self.data, self.index)?)
    }
}

pub fn default_font_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    // macOS
    dirs.push(PathBuf::from("/System/Library/Fonts"));
    dirs.push(PathBuf::from("/Library/Fonts"));
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join("Library/Fonts"));
    }

    // Windows
    if let Some(windir) = std::env::var_os("WINDIR") {
        dirs.push(PathBuf::from(windir).join("Fonts"));
    }
    if let Some(localappdata) = std::env::var_os("LOCALAPPDATA") {
        dirs.push(PathBuf::from(localappdata).join(r"Microsoft\Windows\Fonts"));
    }

    // Linux
    dirs.push(PathBuf::from("/usr/share/fonts"));
    dirs.push(PathBuf::from("/usr/local/share/fonts"));
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join(".local/share/fonts"));
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
    let mut names = HashSet::new();
    for path in discover_font_paths(dirs) {
        let Ok(data) = std::fs::read(&path) else {
            continue;
        };
        let data: Arc<[u8]> = data.into();
        let count = ttf_parser::fonts_in_collection(&data).unwrap_or(1);
        // Reject truncated collection offset arrays before iterating a count
        // taken from an untrusted font header.
        if data.starts_with(b"ttcf") && count as usize > data.len().saturating_sub(12) / 4 {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        for index in 0..count {
            // Keep face zero's existing name for failure-list compatibility.
            let name = if index == 0 {
                stem.to_owned()
            } else {
                format!("{stem}#{index}")
            };
            if let Some(mut fe) = try_load_font(name, path.clone(), Arc::clone(&data), index) {
                let base_name = fe.name.clone();
                let mut suffix = 1;
                while !names.insert(fe.name.clone()) {
                    suffix += 1;
                    fe.name = format!("{base_name}#file{suffix}");
                }
                entries.push(fe);
            }
        }
    }
    entries
}

/// Canonical paths prevent duplicates from overlapping roots and symlink loops.
/// Sorting paths also makes face enumeration independent of filesystem order.
fn discover_font_paths(dirs: &[PathBuf]) -> BTreeSet<PathBuf> {
    let mut paths = BTreeSet::new();
    let mut visited = HashSet::new();
    let mut pending = dirs.to_vec();
    while let Some(dir) = pending.pop() {
        let Ok(dir) = dir.canonicalize() else {
            continue;
        };
        if !visited.insert(dir.clone()) {
            continue;
        }
        let Ok(read_dir) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase());
            match ext.as_deref() {
                Some("ttf" | "otf" | "ttc" | "otc") => {}
                _ => continue,
            }
            if path.is_file()
                && let Ok(path) = path.canonicalize()
            {
                paths.insert(path);
            }
        }
    }
    paths
}

fn try_load_font(name: String, path: PathBuf, data: Arc<[u8]>, index: u32) -> Option<FontEntry> {
    let font_ref = FontRef::try_from_slice_and_index(&data, index).ok()?;
    // Check if font supports Japanese (hiragana 'あ' U+3042)
    let glyph_id = font_ref.glyph_id('あ');
    if glyph_id.0 == 0 {
        return None;
    }
    // Collections can contain different families. Prefer the selected face's
    // typographic/family name over the collection filename when classifying.
    let face = ttf_parser::Face::parse(&data, index).ok()?;
    let family = [
        ttf_parser::name_id::TYPOGRAPHIC_FAMILY,
        ttf_parser::name_id::FAMILY,
    ]
    .into_iter()
    .find_map(|id| {
        face.names()
            .into_iter()
            .filter(|n| n.name_id == id)
            .find_map(|n| n.to_string())
    });
    let style = family
        .map(|family| FontStyle::from_name(&family))
        .filter(|style| *style != FontStyle::Other)
        .unwrap_or_else(|| FontStyle::from_name(&name));
    Some(FontEntry {
        name,
        path,
        data,
        index,
        style,
    })
}

/// A tiny synthetic SFNT fixture with a single cmap entry and rectangle glyphs.
/// No system fonts or third-party font assets are needed by discovery tests.
#[cfg(test)]
pub(crate) fn test_font_bytes(glyph: u32) -> Vec<u8> {
    let mut head = vec![0; 54];
    head[18..20].copy_from_slice(&1000u16.to_be_bytes());
    let mut hhea = vec![0; 36];
    hhea[4..6].copy_from_slice(&800i16.to_be_bytes());
    hhea[6..8].copy_from_slice(&(-200i16).to_be_bytes());
    hhea[34..36].copy_from_slice(&1u16.to_be_bytes());
    let maxp = vec![0, 0, 0x50, 0, 0, 3];
    let hmtx = vec![1, 244, 0, 0, 0, 0, 0, 0];
    let loca = vec![0, 0, 0, 0, 0, 17, 0, 34];
    // One contour: (0,0), (500,0), (500,700), (0,700).
    let mut outline = vec![0, 1, 0, 0, 0, 0, 1, 244, 2, 188, 0, 3, 0, 0, 1, 1, 1, 1];
    for delta in [0i16, 500, 0, -500, 0, 0, 700, 0] {
        outline.extend_from_slice(&delta.to_be_bytes());
    }
    let glyf = [outline.clone(), outline].concat();
    let mut cmap = vec![0, 0, 0, 1, 0, 3, 0, 10, 0, 0, 0, 12, 0, 12, 0, 0];
    for value in [28u32, 0, 1, 0x3042, 0x3042, glyph] {
        cmap.extend_from_slice(&value.to_be_bytes());
    }
    let family = if glyph == 1 {
        "Test Sans-Serif"
    } else {
        "Test Serif"
    };
    let family: Vec<u8> = family.encode_utf16().flat_map(u16::to_be_bytes).collect();
    let mut name = vec![0, 0, 0, 1, 0, 18, 0, 3, 0, 1, 4, 9, 0, 1];
    name.extend_from_slice(&(family.len() as u16).to_be_bytes());
    name.extend_from_slice(&0u16.to_be_bytes());
    name.extend(family);
    let tables = [
        (b"cmap", cmap),
        (b"glyf", glyf),
        (b"head", head),
        (b"hhea", hhea),
        (b"hmtx", hmtx),
        (b"loca", loca),
        (b"maxp", maxp),
        (b"name", name),
    ];
    let mut data = vec![0, 1, 0, 0, 0, 8, 0, 128, 0, 3, 0, 0];
    let mut offset = 12 + tables.len() * 16;
    for (tag, table) in &tables {
        data.extend_from_slice(*tag);
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&(offset as u32).to_be_bytes());
        data.extend_from_slice(&(table.len() as u32).to_be_bytes());
        offset += table.len();
    }
    for (_, table) in tables {
        data.extend(table);
    }
    data
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

    #[test]
    fn styles_prioritize_sans_and_monospace() {
        assert_eq!(FontStyle::from_name("Sans-Serif"), FontStyle::Gothic);
        assert_eq!(
            FontStyle::from_name("Noto Sans Mono CJK"),
            FontStyle::Monospace
        );
        assert_eq!(FontStyle::from_name("Noto Serif CJK"), FontStyle::Mincho);
    }

    #[test]
    fn discovery_recurses_deduplicates_and_sorts() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("nested/fonts");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("Font.ttf"), test_font_bytes(1)).unwrap();
        std::fs::write(dir.path().join("Font.TTF"), test_font_bytes(2)).unwrap();
        std::fs::write(nested.join("broken.otf"), b"not a font").unwrap();
        let roots = [dir.path().to_path_buf(), nested];
        let fonts = discover_fonts(&roots);
        assert_eq!(fonts.len(), 2);
        assert!(fonts[0].path < fonts[1].path);
        assert_ne!(fonts[0].name, fonts[1].name);
        assert_eq!(fonts[0].name, "Font");
        assert_eq!(fonts[1].name, "Font#file2");
    }

    #[cfg(unix)]
    #[test]
    fn discovery_follows_symlinks_without_loops_or_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let font = dir.path().join("Font.ttf");
        std::fs::write(&font, test_font_bytes(1)).unwrap();
        std::os::unix::fs::symlink(&font, dir.path().join("Alias.ttf")).unwrap();
        std::os::unix::fs::symlink(dir.path(), dir.path().join("loop")).unwrap();
        assert_eq!(discover_fonts(&[dir.path().to_path_buf()]).len(), 1);
    }

    #[test]
    fn collection_uses_distinct_faces_and_shares_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let mut data = b"ttcf".to_vec();
        data.extend_from_slice(&0x00010000u32.to_be_bytes());
        data.extend_from_slice(&2u32.to_be_bytes());
        let mut faces = [test_font_bytes(1), test_font_bytes(2)];
        let mut offset = 20u32;
        for face in &mut faces {
            data.extend_from_slice(&offset.to_be_bytes());
            let table_count = u16::from_be_bytes(face[4..6].try_into().unwrap()) as usize;
            for table in face[12..12 + table_count * 16].as_chunks_mut::<16>().0 {
                let local_offset = u32::from_be_bytes(table[8..12].try_into().unwrap());
                table[8..12].copy_from_slice(&(local_offset + offset).to_be_bytes());
            }
            offset += face.len() as u32;
        }
        for face in faces {
            data.extend(face);
        }
        std::fs::write(dir.path().join("Family.otc"), data).unwrap();
        let fonts = discover_fonts(&[dir.path().to_path_buf()]);
        assert_eq!(fonts.len(), 2);
        assert_eq!(fonts[0].name, "Family");
        assert_eq!(fonts[1].name, "Family#1");
        assert_eq!(fonts[0].font_ref().unwrap().glyph_id('あ').0, 1);
        assert_eq!(fonts[1].font_ref().unwrap().glyph_id('あ').0, 2);
        assert_eq!(fonts[0].style, FontStyle::Gothic);
        assert_eq!(fonts[1].style, FontStyle::Mincho);
        assert!(Arc::ptr_eq(&fonts[0].data, &fonts[1].data));
    }

    #[test]
    fn truncated_collection_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut data = b"ttcf".to_vec();
        data.extend_from_slice(&0x00010000u32.to_be_bytes());
        data.extend_from_slice(&u32::MAX.to_be_bytes());
        std::fs::write(dir.path().join("Broken.ttc"), data).unwrap();
        assert!(discover_fonts(&[dir.path().to_path_buf()]).is_empty());
    }
}
