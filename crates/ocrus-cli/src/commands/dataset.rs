use std::path::PathBuf;

use anyhow::Result;
use ocrus_dataset::{
    AugmentConfig, CharFailure, DatasetConfig, FontStyle, generate, generate_from_failures,
};

use super::{DatasetFailuresArgs, DatasetGenerateArgs};

pub fn run_generate(args: &DatasetGenerateArgs) -> Result<()> {
    let categories: Vec<String> = args
        .categories
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let font_styles: Option<Vec<FontStyle>> = args.font_styles.as_ref().map(|s| {
        s.split(',')
            .filter_map(|style| match style.trim().to_lowercase().as_str() {
                "mincho" => Some(FontStyle::Mincho),
                "gothic" => Some(FontStyle::Gothic),
                "script" => Some(FontStyle::Script),
                "monospace" => Some(FontStyle::Monospace),
                "other" => Some(FontStyle::Other),
                _ => {
                    eprintln!("Warning: unknown font style '{style}', skipping");
                    None
                }
            })
            .collect()
    });

    let config = DatasetConfig {
        output_dir: args.output.clone(),
        font_dirs: default_font_dirs(),
        char_data_dir: args
            .char_data_dir
            .clone()
            .unwrap_or_else(default_char_data_dir),
        categories,
        chars_per_image: args.chars_per_image,
        augment: AugmentConfig::default(),
        samples_per_char: args.samples_per_char,
        val_ratio: args.val_ratio,
        font_styles,
    };

    let stats = generate(&config)?;
    println!(
        "Generated {} images ({} train, {} val) in {:.1}s",
        stats.total_images,
        stats.train_images,
        stats.val_images,
        stats.elapsed.as_secs_f64()
    );
    for (cat, count) in &stats.per_category {
        println!("  {cat}: {count} images");
    }
    for (font, count) in &stats.per_font {
        println!("  {font}: {count} images");
    }
    Ok(())
}

pub fn run_from_failures(args: &DatasetFailuresArgs) -> Result<()> {
    let failures_json = std::fs::read_to_string(&args.failures)?;
    let failures: Vec<CharFailure> = serde_json::from_str(&failures_json)?;
    println!(
        "Loaded {} failures from {}",
        failures.len(),
        args.failures.display()
    );

    let config = DatasetConfig {
        output_dir: args.output.clone(),
        font_dirs: default_font_dirs(),
        char_data_dir: default_char_data_dir(),
        categories: vec![],
        samples_per_char: args.samples_per_char,
        ..Default::default()
    };

    let stats = generate_from_failures(&failures, &config)?;
    println!(
        "Generated {} images ({} train, {} val) in {:.1}s",
        stats.total_images,
        stats.train_images,
        stats.val_images,
        stats.elapsed.as_secs_f64()
    );
    Ok(())
}

fn default_font_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![
        PathBuf::from("/System/Library/Fonts"),
        PathBuf::from("/Library/Fonts"),
    ];
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(PathBuf::from(home).join("Library/Fonts"));
    }
    dirs
}

fn default_char_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/test_chars")
}
