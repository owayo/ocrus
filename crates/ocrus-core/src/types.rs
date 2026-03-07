use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl BBox {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn area(&self) -> u32 {
        self.width * self.height
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quad {
    pub points: [(f32, f32); 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLine {
    pub text: String,
    pub bbox: BBox,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page {
    pub width: u32,
    pub height: u32,
    pub lines: Vec<TextLine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    pub pages: Vec<Page>,
}

impl OcrResult {
    pub fn full_text(&self) -> String {
        self.pages
            .iter()
            .flat_map(|p| p.lines.iter().map(|l| l.text.as_str()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
