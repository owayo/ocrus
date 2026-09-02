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
pub use font::{FontEntry, FontStyle};
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
    pub font_styles: Option<Vec<font::FontStyle>>,
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
            font_styles: None,
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
    let fonts = font::discover_fonts_filtered(&config.font_dirs, config.font_styles.as_deref());
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

/// Generate training data from pre-rendered test images (no font rendering needed).
///
/// Looks up images at `test_images_dir/{font_name}/{category}/U+{codepoint}.png`
/// for each failure entry, applies augmentations, and writes to the output directory.
pub fn generate_from_failure_images(
    failures: &[CharFailure],
    test_images_dir: &Path,
    config: &DatasetConfig,
) -> Result<DatasetStats> {
    use image::ImageReader;

    let start = Instant::now();
    let dataset_writer = writer::DatasetWriter::new(&config.output_dir)?;

    // Deduplicate failures by (character, font_name, category)
    let unique_failures: Vec<&CharFailure> = {
        let mut seen = std::collections::HashSet::new();
        failures
            .iter()
            .filter(|f| {
                let key = (f.character, f.font_name.clone(), f.category.clone());
                seen.insert(key)
            })
            .collect()
    };

    // Group by font_name for efficient directory traversal
    let by_font: std::collections::HashMap<String, Vec<&CharFailure>> = {
        let mut map: std::collections::HashMap<String, Vec<&CharFailure>> =
            std::collections::HashMap::new();
        for f in &unique_failures {
            let font = f.font_name.clone().unwrap_or_default();
            map.entry(font).or_default().push(f);
        }
        map
    };

    let skipped_atomic = std::sync::atomic::AtomicU64::new(0);

    by_font
        .par_iter()
        .try_for_each(|(font_name, font_failures)| -> Result<()> {
            let mut rng = rand::rng();

            for failure in font_failures {
                let codepoint = failure.character as u32;
                let image_filename = format!("U+{codepoint:04X}.png");

                // Build path: test_images_dir/{font_name}/{category}/U+XXXX.png
                let image_path = if font_name.is_empty() {
                    // No font_name: skip (shouldn't happen with test_images)
                    skipped_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    continue;
                } else {
                    test_images_dir
                        .join(font_name)
                        .join(&failure.category)
                        .join(&image_filename)
                };

                let img = match ImageReader::open(&image_path) {
                    Ok(reader) => match reader.decode() {
                        Ok(dynimg) => dynimg.to_luma8(),
                        Err(e) => {
                            eprintln!("Warning: failed to decode {}: {e}", image_path.display());
                            skipped_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            continue;
                        }
                    },
                    Err(_) => {
                        skipped_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        continue;
                    }
                };

                let actual_font = failure.font_name.as_deref().unwrap_or("unknown");

                for aug in &config.augment.types {
                    let augmented = augment::apply_augmentation(&img, aug, &mut rng);
                    dataset_writer.add_sample(
                        &augmented,
                        &failure.character.to_string(),
                        &failure.category,
                        actual_font,
                        &aug.label(),
                    )?;
                }
            }
            Ok(())
        })?;

    let skipped = skipped_atomic.load(std::sync::atomic::Ordering::Relaxed);
    if skipped > 0 {
        eprintln!("Warning: skipped {skipped} failures (image not found)");
    }

    let found = unique_failures.len() as u64 - skipped;
    println!(
        "Processed {found} failure images ({} augmentations each)",
        config.augment.types.len()
    );

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
