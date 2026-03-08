use std::path::PathBuf;

use ab_glyph::{Font, FontRef, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_text_mut, text_size};
use serde::Deserialize;

const FONT_SIZE: f32 = 48.0;
const PADDING: u32 = 20;

#[derive(Deserialize)]
struct FontListConfig {
    fonts: Vec<FontConfig>,
}

#[derive(Deserialize)]
struct FontConfig {
    name: String,
    file: String,
    #[serde(default)]
    index: u32,
}

/// Generate test images for E2E accuracy testing.
///
/// For each font x category x character, renders a single character as a PNG image
/// and saves it to `test_images/{font_name}/{category}/U+{XXXX}.png`.
#[test]
#[ignore]
fn generate_test_images() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    // Load font config
    let config_path = workspace_root.join("test_fonts.yml");
    let config_content = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", config_path.display()));
    let config: FontListConfig =
        serde_yaml::from_str(&config_content).expect("Failed to parse test_fonts.yml");

    let fonts_dir = workspace_root.join("fonts");
    assert!(
        fonts_dir.is_dir(),
        "fonts/ directory not found at {}",
        fonts_dir.display()
    );

    // Discover categories from data/test_chars/*.txt
    let test_chars_dir = workspace_root.join("data/test_chars");
    assert!(
        test_chars_dir.is_dir(),
        "data/test_chars/ not found at {}",
        test_chars_dir.display()
    );

    let mut categories: Vec<String> = std::fs::read_dir(&test_chars_dir)
        .expect("Failed to read data/test_chars/")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".txt") {
                Some(name.trim_end_matches(".txt").to_string())
            } else {
                None
            }
        })
        .collect();
    categories.sort();

    println!(
        "Found {} categories: {}",
        categories.len(),
        categories.join(", ")
    );

    let output_base = workspace_root.join("test_images");
    let mut total_generated = 0u64;
    let mut total_skipped = 0u64;

    for fc in &config.fonts {
        let font_path = fonts_dir.join(&fc.file);
        if !font_path.exists() {
            println!("SKIP font not found: {} ({})", fc.name, font_path.display());
            continue;
        }

        let font_data = match std::fs::read(&font_path) {
            Ok(data) => data,
            Err(e) => {
                println!("SKIP failed to read font {}: {e}", fc.name);
                continue;
            }
        };

        let font = match FontRef::try_from_slice_and_index(&font_data, fc.index) {
            Ok(f) => f,
            Err(e) => {
                println!("SKIP failed to parse font {}: {e}", fc.name);
                continue;
            }
        };

        println!("=== Font: {} ===", fc.name);

        for category in &categories {
            let chars_path = test_chars_dir.join(format!("{category}.txt"));
            let content = match std::fs::read_to_string(&chars_path) {
                Ok(c) => c,
                Err(_) => {
                    println!("  {category}: (no data)");
                    continue;
                }
            };
            let chars: Vec<char> = content.chars().filter(|c| !c.is_whitespace()).collect();
            if chars.is_empty() {
                println!("  {category}: (empty)");
                continue;
            }

            let out_dir = output_base.join(&fc.name).join(category);
            std::fs::create_dir_all(&out_dir)
                .unwrap_or_else(|e| panic!("Failed to create {}: {e}", out_dir.display()));

            let mut cat_generated = 0u64;
            let mut cat_skipped = 0u64;

            for &ch in &chars {
                let scale = PxScale::from(FONT_SIZE);

                // Skip characters the font cannot render (zero-size glyph)
                let glyph_id = font.glyph_id(ch);
                if let Some(outlined) = font.outline_glyph(glyph_id.with_scale(scale)) {
                    let bounds = outlined.px_bounds();
                    if bounds.width() < 1.0 || bounds.height() < 1.0 {
                        cat_skipped += 1;
                        continue;
                    }
                } else {
                    cat_skipped += 1;
                    continue;
                }

                let text = ch.to_string();
                let (text_w, text_h) = text_size(scale, &font, &text);
                if text_w == 0 || text_h == 0 {
                    cat_skipped += 1;
                    continue;
                }

                let img_w = text_w + PADDING * 2;
                let img_h = text_h + PADDING * 2;
                let mut img = RgbImage::from_pixel(img_w, img_h, Rgb([255u8, 255, 255]));
                draw_text_mut(
                    &mut img,
                    Rgb([0u8, 0, 0]),
                    PADDING as i32,
                    PADDING as i32,
                    scale,
                    &font,
                    &text,
                );

                let codepoint = ch as u32;
                let filename = format!("U+{codepoint:04X}.png");
                let out_path = out_dir.join(&filename);
                img.save(&out_path)
                    .unwrap_or_else(|e| panic!("Failed to save {}: {e}", out_path.display()));

                cat_generated += 1;
            }

            println!(
                "  {category}: generated {cat_generated}, skipped {cat_skipped} (total chars: {})",
                chars.len()
            );
            total_generated += cat_generated;
            total_skipped += cat_skipped;
        }
    }

    println!("=== Summary ===");
    println!("Total images generated: {total_generated}");
    println!("Total characters skipped: {total_skipped}");
    println!("Output directory: {}", output_base.display());
}
