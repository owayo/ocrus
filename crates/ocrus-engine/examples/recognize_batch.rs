//! Recognize a JSON array of image paths from stdin with one production engine.
//! Usage: recognize_batch MODEL_DIR THREADS < images.json

use std::error::Error;
use std::io;
use std::path::PathBuf;

use ocrus_core::EngineConfigBuilder;
use ocrus_engine::OcrEngine;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = PathBuf::from(args.next().ok_or("missing model directory")?);
    let threads: usize = args
        .next()
        .ok_or("missing thread count")?
        .to_string_lossy()
        .parse()?;
    if threads == 0 || args.next().is_some() {
        return Err("expected model directory and positive thread count".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let engine = OcrEngine::new(
        EngineConfigBuilder::new()
            .model_dir(model_dir)
            .num_threads(threads)
            .build(),
    )?;
    let paths: Vec<PathBuf> = serde_json::from_reader(io::stdin().lock())?;
    let results = paths
        .iter()
        .map(|path| engine.recognize_path(path))
        .collect::<ocrus_core::error::Result<Vec<_>>>()?;
    serde_json::to_writer(io::stdout().lock(), &results)?;
    Ok(())
}
