pub mod augment;
pub mod charsets;
pub mod font;
pub mod render;
pub mod writer;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, bail};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub use augment::AugmentType;
pub use font::FontEntry;
pub use writer::DatasetStats;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentConfig {
    pub types: Vec<AugmentType>,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            types: vec![
                AugmentType::Original,
                AugmentType::Rotate(2.0),
                AugmentType::Blur(1.0),
                AugmentType::Noise(0.02),
                AugmentType::Contrast(1.1),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub char_data_dir: PathBuf,
    pub font_dirs: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub categories: Vec<String>,
    pub chars_per_image: usize,
    pub augment: AugmentConfig,
    pub samples_per_char: usize,
    pub val_ratio: f32,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            char_data_dir: PathBuf::from("data/test_chars"),
            font_dirs: font::default_font_dirs(),
            output_dir: PathBuf::from("output/dataset"),
            categories: vec!["hiragana".to_string(), "katakana".to_string()],
            chars_per_image: 1,
            augment: AugmentConfig::default(),
            samples_per_char: 5,
            val_ratio: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharFailure {
    pub character: char,
    pub category: String,
    pub font_name: Option<String>,
}

const RENDER_HEIGHT: u32 = 48;

pub fn generate(config: &DatasetConfig) -> Result<DatasetStats> {
    let start = Instant::now();
    let fonts = font::discover_fonts(&config.font_dirs);
    if fonts.is_empty() {
        bail!("no Japanese-capable fonts found in {:?}", config.font_dirs);
    }

    let mut all_chars: Vec<(String, Vec<char>)> = Vec::new();
    for cat in &config.categories {
        let chars = charsets::load_charset(&config.char_data_dir, cat)?;
        all_chars.push((cat.clone(), chars));
    }

    let dataset_writer = writer::DatasetWriter::new(&config.output_dir)?;

    fonts.par_iter().try_for_each(|font_entry| -> Result<()> {
        let font_ref = font_entry.font_ref()?;
        let mut rng = rand::rng();

        for (category, chars) in &all_chars {
            for chunk in chars.chunks(config.chars_per_image) {
                let text: String = chunk.iter().collect();
                for _ in 0..config.samples_per_char {
                    let img = render::render_text_line(&font_ref, &text, RENDER_HEIGHT);
                    for aug in &config.augment.types {
                        let augmented = augment::apply_augmentation(&img, aug, &mut rng);
                        dataset_writer.add_sample(
                            &augmented,
                            &text,
                            category,
                            &font_entry.name,
                            &aug.label(),
                        )?;
                    }
                }
            }
        }
        Ok(())
    })?;

    let mut stats = dataset_writer.finish(config.val_ratio)?;
    stats.elapsed = start.elapsed();
    Ok(stats)
}

pub fn generate_from_failures(
    failures: &[CharFailure],
    config: &DatasetConfig,
) -> Result<DatasetStats> {
    let start = Instant::now();
    let fonts = font::discover_fonts(&config.font_dirs);
    if fonts.is_empty() {
        bail!("no Japanese-capable fonts found in {:?}", config.font_dirs);
    }

    let dataset_writer = writer::DatasetWriter::new(&config.output_dir)?;

    fonts.par_iter().try_for_each(|font_entry| -> Result<()> {
        let font_ref = font_entry.font_ref()?;
        let mut rng = rand::rng();

        for failure in failures {
            if let Some(ref fname) = failure.font_name
                && &font_entry.name != fname
            {
                continue;
            }

            for _ in 0..config.samples_per_char {
                let img = render::render_text_line(
                    &font_ref,
                    &failure.character.to_string(),
                    RENDER_HEIGHT,
                );
                for aug in &config.augment.types {
                    let augmented = augment::apply_augmentation(&img, aug, &mut rng);
                    dataset_writer.add_sample(
                        &augmented,
                        &failure.character.to_string(),
                        &failure.category,
                        &font_entry.name,
                        &aug.label(),
                    )?;
                }
            }
        }
        Ok(())
    })?;

    let mut stats = dataset_writer.finish(config.val_ratio)?;
    stats.elapsed = start.elapsed();
    Ok(stats)
}

pub fn available_categories(data_dir: &Path) -> Result<Vec<String>> {
    let mut categories = Vec::new();
    for entry in std::fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "txt")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
        {
            categories.push(stem.to_string());
        }
    }
    categories.sort();
    Ok(categories)
}
