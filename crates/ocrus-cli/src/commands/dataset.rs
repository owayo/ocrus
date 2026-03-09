use std::path::PathBuf;

use anyhow::Result;
use ocrus_dataset::{
    AugmentConfig, CharFailure, DatasetConfig, FontStyle, generate, generate_from_failure_images,
    generate_from_failures,
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
    let mut failures: Vec<CharFailure> = serde_json::from_str(&failures_json)?;
    println!(
        "Loaded {} failures from {}",
        failures.len(),
        args.failures.display()
    );

    if args.all_fonts {
        for f in &mut failures {
            f.font_name = None;
        }
        println!("  --all-fonts: ignoring font_name, generating with all available fonts");
    }

    let mut font_dirs = default_font_dirs();
    font_dirs.extend(args.font_dirs.iter().cloned());

    let config = DatasetConfig {
        output_dir: args.output.clone(),
        font_dirs,
        char_data_dir: default_char_data_dir(),
        categories: vec![],
        samples_per_char: args.samples_per_char,
        ..Default::default()
    };

    let stats = if let Some(ref test_images_dir) = args.test_images {
        println!(
            "Using pre-rendered test images from {}",
            test_images_dir.display()
        );
        generate_from_failure_images(&failures, test_images_dir, &config)?
    } else {
        generate_from_failures(&failures, &config)?
    };
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
    ocrus_dataset::font::default_font_dirs()
}

fn default_char_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/test_chars")
}
