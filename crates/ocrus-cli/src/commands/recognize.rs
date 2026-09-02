use anyhow::{Context, Result};

use ocrus_core::{CharsetMode, EngineConfig, EngineConfigBuilder, OcrMode, OcrResult};
use ocrus_engine::OcrEngine;

use super::{CliCharset, CliMode, OutputFormat, RecognizeArgs};
use crate::output;

pub fn run(args: RecognizeArgs) -> Result<()> {
    let engine =
        OcrEngine::new(build_engine_config(&args)).context("Failed to build OCR engine")?;

    let result = engine
        .recognize_path(&args.input)
        .with_context(|| format!("Failed to recognize {}", args.input.display()))?;

    output_result(&result, &args.format)
}

fn build_engine_config(args: &RecognizeArgs) -> EngineConfig {
    let mut builder = EngineConfigBuilder::new();

    if let Some(ref dir) = args.model_dir {
        builder = builder.model_dir(dir.clone());
    }
    if let Some(threads) = args.threads {
        builder = builder.num_threads(threads);
    }

    builder = builder.mode(match args.mode {
        CliMode::Auto => OcrMode::Auto,
        CliMode::Fastest => OcrMode::Fastest,
        CliMode::Accurate => OcrMode::Accurate,
    });

    builder = builder.charset(match args.charset {
        CliCharset::Full => CharsetMode::Full,
        CliCharset::Jis => CharsetMode::Jis,
    });

    if let Some(ref path) = args.dict {
        builder = builder.dict_path(path.clone());
    }

    builder = builder.ruby_separation(args.ruby);

    if let Some(ref path) = args.cascade {
        builder = builder.cascade_model_path(path.clone());
    }

    builder.build()
}

fn output_result(result: &OcrResult, format: &OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Text => output::text::print(result),
        OutputFormat::Json => output::json::print(result),
    }
}
