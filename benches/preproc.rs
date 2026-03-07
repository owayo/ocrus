// Criterion benchmark for preprocessing + layout + recognizer pipeline.
//
// [[bench]]
// name = "preproc"
// harness = false

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array2;

/// Generate a synthetic binary image with text-like horizontal lines.
fn generate_test_image(height: usize, width: usize, num_lines: usize) -> Array2<u8> {
    let mut img = Array2::from_elem((height, width), 255u8);
    let line_spacing = height / (num_lines + 1);
    let line_height = (line_spacing as f32 * 0.4) as usize;

    for i in 0..num_lines {
        let y_start = line_spacing * (i + 1) - line_height / 2;
        let y_end = (y_start + line_height).min(height);
        for y in y_start..y_end {
            for x in (width / 20)..(width * 19 / 20) {
                img[[y, x]] = 0;
            }
        }
    }
    img
}

/// Generate a grayscale image with text-like dark stripes on light background.
fn generate_gray_image(width: usize, height: usize) -> Array2<u8> {
    let mut img = Array2::from_elem((height, width), 200u8);
    let line_spacing = height / 6;
    for i in 0..5 {
        let y_start = line_spacing * (i + 1) - 5;
        let y_end = (y_start + 10).min(height);
        for y in y_start..y_end {
            for x in (width / 10)..(width * 9 / 10) {
                img[[y, x]] = 30;
            }
        }
    }
    img
}

fn bench_grayscale(c: &mut Criterion) {
    use image::{DynamicImage, RgbImage};
    let rgb = RgbImage::from_fn(800, 600, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });
    let img = DynamicImage::ImageRgb8(rgb);

    c.bench_function("grayscale_800x600", |b| {
        b.iter(|| {
            let gray = ocrus_preproc::to_grayscale(black_box(&img));
            black_box(gray);
        });
    });
}

fn bench_binarize(c: &mut Criterion) {
    let gray = generate_gray_image(800, 600);

    c.bench_function("binarize_otsu_800x600", |b| {
        b.iter(|| black_box(ocrus_preproc::binarize_otsu(black_box(&gray))))
    });

    c.bench_function("binarize_sauvola_800x600", |b| {
        b.iter(|| black_box(ocrus_preproc::binarize_sauvola(black_box(&gray), 15, 0.2)))
    });

    c.bench_function("binarize_adaptive_800x600", |b| {
        b.iter(|| black_box(ocrus_preproc::binarize_adaptive(black_box(&gray))))
    });
}

fn bench_layout_detection(c: &mut Criterion) {
    let binary = generate_test_image(600, 800, 5);

    c.bench_function("layout_detect_lines_600x800", |b| {
        b.iter(|| {
            let lines = ocrus_layout::detect_lines_projection(black_box(&binary));
            black_box(lines);
        });
    });
}

fn bench_ccl_detection(c: &mut Criterion) {
    let binary = generate_test_image(600, 800, 5);

    c.bench_function("ccl_detect_lines_600x800", |b| {
        b.iter(|| {
            let lines = ocrus_layout::detect_lines_ccl(black_box(&binary));
            black_box(lines);
        });
    });
}

fn bench_vertical_detection(c: &mut Criterion) {
    let binary = generate_test_image(800, 600, 5);

    c.bench_function("orientation_detect_800x600", |b| {
        b.iter(|| {
            let orient = ocrus_layout::detect_orientation(black_box(&binary));
            black_box(orient);
        });
    });
}

fn bench_quality_assessment(c: &mut Criterion) {
    let binary = generate_test_image(600, 800, 5);

    c.bench_function("quality_assess_600x800", |b| {
        b.iter(|| {
            let quality = ocrus_layout::assess_quality(black_box(&binary));
            black_box(quality);
        });
    });
}

fn bench_layout_large(c: &mut Criterion) {
    let binary = generate_test_image(2000, 3000, 20);

    c.bench_function("layout_detect_lines_2000x3000", |b| {
        b.iter(|| {
            let lines = ocrus_layout::detect_lines_projection(black_box(&binary));
            black_box(lines);
        });
    });
}

fn bench_ctc_decode(c: &mut Criterion) {
    use ocrus_recognizer::charset::Charset;

    let charset = Charset::from_chars(
        &(0..100)
            .map(|i| char::from_u32(0x3040 + i).unwrap_or('?'))
            .collect::<Vec<_>>(),
    );
    // 100 timesteps, 101 classes (blank + 100 chars)
    let logits: Vec<f32> = (0..100 * 101)
        .map(|i| ((i % 101) as f32 * 0.1) - 5.0)
        .collect();

    c.bench_function("ctc_greedy_100ts_101cls", |b| {
        b.iter(|| {
            let (text, conf) = ocrus_recognizer::ctc_greedy_decode(
                black_box(&logits),
                100,
                101,
                black_box(&charset),
            );
            black_box((text, conf));
        });
    });

    c.bench_function("ctc_beam_100ts_101cls_w5", |b| {
        b.iter(|| {
            let (text, conf) = ocrus_recognizer::ctc_beam_decode(
                black_box(&logits),
                100,
                101,
                black_box(&charset),
                5,
            );
            black_box((text, conf));
        });
    });
}

criterion_group!(
    benches,
    bench_grayscale,
    bench_binarize,
    bench_layout_detection,
    bench_ccl_detection,
    bench_vertical_detection,
    bench_quality_assessment,
    bench_layout_large,
    bench_ctc_decode,
);
criterion_main!(benches);
