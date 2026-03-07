use std::time::Instant;

use anyhow::{Context, Result};

use ocrus_layout::detect_lines_projection;
use ocrus_preproc::{binarize_otsu, to_grayscale};

use super::BenchArgs;

pub fn run(args: BenchArgs) -> Result<()> {
    let img = image::open(&args.input)
        .with_context(|| format!("Failed to open image: {}", args.input.display()))?;

    println!("Benchmarking: {}", args.input.display());
    println!("Image size: {}x{}", img.width(), img.height());
    println!("Iterations: {}", args.iterations);
    println!();

    // Benchmark preprocessing
    let start = Instant::now();
    for _ in 0..args.iterations {
        let gray = to_grayscale(&img);
        std::hint::black_box(&gray);
    }
    let preproc_time = start.elapsed() / args.iterations;
    println!("Preprocessing (grayscale): {:?}", preproc_time);

    // Benchmark binarization
    let gray = to_grayscale(&img);
    let start = Instant::now();
    for _ in 0..args.iterations {
        let bin = binarize_otsu(&gray);
        std::hint::black_box(&bin);
    }
    let binarize_time = start.elapsed() / args.iterations;
    println!("Binarization (Otsu):       {:?}", binarize_time);

    // Benchmark layout
    let binary = binarize_otsu(&gray);
    let start = Instant::now();
    for _ in 0..args.iterations {
        let lines = detect_lines_projection(&binary);
        std::hint::black_box(&lines);
    }
    let layout_time = start.elapsed() / args.iterations;
    println!("Layout (projection):       {:?}", layout_time);

    println!();
    println!(
        "Total (preproc+layout):    {:?}",
        preproc_time + binarize_time + layout_time
    );

    Ok(())
}
