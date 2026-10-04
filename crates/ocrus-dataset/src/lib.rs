pub mod augment;
pub mod charsets;
pub mod font;
pub mod render;
pub mod writer;

use std::path::{Path, PathBuf};
use std::time::Instant;

use ab_glyph::Font;
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
    /// Native canvas heights. Training preprocessing resizes the saved images.
    #[serde(default = "default_render_heights")]
    pub render_heights: Vec<u32>,
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
            render_heights: default_render_heights(),
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

fn default_render_heights() -> Vec<u32> {
    vec![RENDER_HEIGHT]
}

fn validated_render_heights(config: &DatasetConfig) -> Result<Vec<u32>> {
    anyhow::ensure!(
        !config.render_heights.is_empty() && config.render_heights.iter().all(|&h| h > 0),
        "render_heights must contain at least one height, and all heights must be greater than zero"
    );
    let mut heights = config.render_heights.clone();
    heights.sort_unstable();
    heights.dedup();
    Ok(heights)
}

fn augmentation_label(height: u32, aug: &AugmentType) -> String {
    if height == RENDER_HEIGHT {
        aug.label()
    } else {
        format!("render_{height}px_{}", aug.label())
    }
}

pub fn generate(config: &DatasetConfig) -> Result<DatasetStats> {
    anyhow::ensure!(
        config.chars_per_image > 0,
        "chars_per_image must be greater than zero"
    );
    writer::validate_val_ratio(config.val_ratio)?;
    let render_heights = validated_render_heights(config)?;
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
            // Filter before grouping, so missing glyphs cannot become tofu
            // images carrying a label for a character the font did not draw.
            let supported: Vec<char> = chars
                .iter()
                .copied()
                .filter(|&ch| font_ref.glyph_id(ch).0 != 0)
                .collect();
            for chunk in supported.chunks(config.chars_per_image) {
                let text: String = chunk.iter().collect();
                for &height in &render_heights {
                    let img = render::render_text_line(&font_ref, &text, height);
                    for _ in 0..config.samples_per_char {
                        for aug in &config.augment.types {
                            let augmented = augment::apply_augmentation(&img, aug, &mut rng);
                            dataset_writer.add_sample(
                                &augmented,
                                &text,
                                category,
                                &font_entry.name,
                                &augmentation_label(height, aug),
                            )?;
                        }
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
    writer::validate_val_ratio(config.val_ratio)?;
    let render_heights = validated_render_heights(config)?;
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
            if font_ref.glyph_id(failure.character).0 == 0 {
                continue;
            }

            for &height in &render_heights {
                let img =
                    render::render_text_line(&font_ref, &failure.character.to_string(), height);
                for _ in 0..config.samples_per_char {
                    for aug in &config.augment.types {
                        let augmented = augment::apply_augmentation(&img, aug, &mut rng);
                        dataset_writer.add_sample(
                            &augmented,
                            &failure.character.to_string(),
                            &failure.category,
                            &font_entry.name,
                            &augmentation_label(height, aug),
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

/// 描画済みのテスト画像から学習データを生成する。
///
/// `test_images_dir/{font_name}/{category}/U+{codepoint}.png` を読み、
/// 拡張処理した画像を出力ディレクトリに保存する。
pub fn generate_from_failure_images(
    failures: &[CharFailure],
    test_images_dir: &Path,
    config: &DatasetConfig,
) -> Result<DatasetStats> {
    use image::ImageReader;

    writer::validate_val_ratio(config.val_ratio)?;
    let start = Instant::now();
    let dataset_writer = writer::DatasetWriter::new(&config.output_dir)?;

    // 文字・フォント・カテゴリが同じ失敗例をまとめる。
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

    // ディレクトリの探索回数を減らすため、フォント別にまとめる。
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

                // フォント・カテゴリ・コードポイントから画像パスを組み立てる。
                let image_path = if font_name.is_empty() {
                    // フォント名がない失敗例は画像を特定できないため飛ばす。
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_text_images_keep_native_heights_and_deduplicate_sizes() {
        let dir = tempfile::tempdir().unwrap();
        let fonts = dir.path().join("fonts");
        let chars = dir.path().join("chars");
        std::fs::create_dir_all(&fonts).unwrap();
        std::fs::create_dir_all(&chars).unwrap();
        std::fs::write(fonts.join("Test.ttf"), font::test_font_bytes(1)).unwrap();
        std::fs::write(chars.join("test.txt"), "あ").unwrap();
        let mut config = DatasetConfig {
            font_dirs: vec![fonts],
            char_data_dir: chars,
            output_dir: dir.path().join("generated"),
            categories: vec!["test".into()],
            samples_per_char: 1,
            render_heights: vec![48, 12, 12],
            augment: AugmentConfig {
                types: vec![AugmentType::Original],
            },
            ..DatasetConfig::default()
        };
        let failure = CharFailure {
            character: 'あ',
            category: "test".into(),
            font_name: None,
        };
        for from_failures in [false, true] {
            config.output_dir = dir.path().join(if from_failures {
                "failures"
            } else {
                "generated"
            });
            let stats = if from_failures {
                generate_from_failures(std::slice::from_ref(&failure), &config).unwrap()
            } else {
                generate(&config).unwrap()
            };
            assert_eq!(stats.total_images, 2);
            let labels = std::fs::read_to_string(config.output_dir.join("labels.tsv")).unwrap();
            let mut heights = Vec::new();
            for row in labels.lines().skip(1) {
                let columns: Vec<_> = row.split('\t').collect();
                let image = image::open(config.output_dir.join("samples").join(columns[0]))
                    .unwrap()
                    .to_luma8();
                let height = match columns[4] {
                    "original" => 48,
                    "render_12px_original" => 12,
                    other => panic!("unexpected augmentation: {other}"),
                };
                assert_eq!(image.height(), height);
                assert!(image.pixels().any(|p| p[0] < 255));
                heights.push(height);
            }
            heights.sort_unstable();
            assert_eq!(heights, [12, 48]);
        }
    }

    #[test]
    fn font_generation_rejects_empty_or_zero_render_heights_before_loading_fonts() {
        for render_heights in [vec![], vec![0], vec![48, 0]] {
            let config = DatasetConfig {
                font_dirs: vec![],
                render_heights,
                ..DatasetConfig::default()
            };
            for result in [generate(&config), generate_from_failures(&[], &config)] {
                assert!(result.unwrap_err().to_string().contains("render_heights"));
            }
        }
    }

    #[test]
    fn older_dataset_config_defaults_to_48px_rendering() {
        let mut json = serde_json::to_value(DatasetConfig::default()).unwrap();
        json.as_object_mut().unwrap().remove("render_heights");
        let config: DatasetConfig = serde_json::from_value(json).unwrap();
        assert_eq!(config.render_heights, [48]);
    }

    #[test]
    fn missing_glyphs_never_receive_training_labels() {
        let dir = tempfile::tempdir().unwrap();
        let fonts = dir.path().join("fonts");
        let chars = dir.path().join("chars");
        std::fs::create_dir_all(&fonts).unwrap();
        std::fs::create_dir_all(&chars).unwrap();
        std::fs::write(fonts.join("Test.ttf"), font::test_font_bytes(1)).unwrap();
        std::fs::write(chars.join("test.txt"), "あ漢").unwrap();
        let mut config = DatasetConfig {
            font_dirs: vec![fonts],
            char_data_dir: chars,
            output_dir: dir.path().join("generated"),
            categories: vec!["test".into()],
            chars_per_image: 2,
            samples_per_char: 1,
            augment: AugmentConfig {
                types: vec![AugmentType::Original],
            },
            ..DatasetConfig::default()
        };
        let stats = generate(&config).unwrap();
        assert_eq!(stats.total_images, 1);
        let labels = std::fs::read_to_string(config.output_dir.join("labels.tsv")).unwrap();
        assert_eq!(
            labels.lines().nth(1).unwrap().split('\t').nth(1),
            Some("あ")
        );

        config.output_dir = dir.path().join("failures");
        let failures = ['あ', '漢'].map(|character| CharFailure {
            character,
            category: "test".into(),
            font_name: None,
        });
        let stats = generate_from_failures(&failures, &config).unwrap();
        assert_eq!(stats.total_images, 1);
        let labels = std::fs::read_to_string(config.output_dir.join("labels.tsv")).unwrap();
        assert_eq!(
            labels.lines().nth(1).unwrap().split('\t').nth(1),
            Some("あ")
        );
    }

    #[test]
    fn generation_validates_before_loading_fonts_or_creating_output() {
        let mut config = DatasetConfig {
            chars_per_image: 0,
            font_dirs: vec![],
            ..DatasetConfig::default()
        };
        assert!(
            generate(&config)
                .unwrap_err()
                .to_string()
                .contains("chars_per_image")
        );
        config.chars_per_image = 1;
        config.val_ratio = f32::NAN;
        for result in [
            generate(&config),
            generate_from_failures(&[], &config),
            generate_from_failure_images(&[], Path::new("missing-images"), &config),
        ] {
            assert!(result.unwrap_err().to_string().contains("validation ratio"));
        }
    }
}
