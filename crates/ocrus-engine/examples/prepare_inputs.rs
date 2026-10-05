//! Export production inputs for training without loading a recognition model.
//! Usage: cargo run -p ocrus-engine --release --example prepare_inputs -- manifest.json output-dir
//! The manifest is an array of image paths. Each input is saved as little-endian f32.

use std::error::Error;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use ocrus_core::EngineConfig;
use ocrus_engine::prepare_image;
use serde_json::json;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let manifest = PathBuf::from(args.next().ok_or("missing manifest.json")?);
    let destination = PathBuf::from(args.next().ok_or("missing output directory")?);
    if args.next().is_some() {
        return Err("expected manifest.json and output directory".into());
    }
    fs::create_dir(&destination)?;
    let paths: Vec<PathBuf> = serde_json::from_slice(&fs::read(manifest)?)?;
    let config = EngineConfig::default();
    let mut records = Vec::with_capacity(paths.len());
    for (index, path) in paths.iter().enumerate() {
        let image = image::open(path)?;
        let prepared = prepare_image(&image, &config);
        let mut lines = Vec::with_capacity(prepared.lines.len());
        for (line_index, line) in prepared.lines.iter().enumerate() {
            let input = &line.inputs[0];
            let filename = format!("{index:06}-{line_index:03}.f32");
            let mut writer = BufWriter::new(fs::File::create(destination.join(&filename))?);
            for value in &input.data {
                writer.write_all(&value.to_le_bytes())?;
            }
            writer.flush()?;
            lines.push(json!({"file": filename, "shape": input.shape, "bbox": line.bbox}));
        }
        records.push(json!({"image": path, "width": prepared.width,
            "height": prepared.height, "lines": lines}));
        if index % 100 == 0 {
            eprintln!("Prepared {}/{} images", index + 1, paths.len());
        }
    }
    fs::write(
        destination.join("manifest.json"),
        serde_json::to_vec_pretty(&records)?,
    )?;
    Ok(())
}
