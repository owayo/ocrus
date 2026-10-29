use ndarray::Array2;
use ocrus_core::BBox;
use wide::u8x16;

/// 水平射影で文字行を検出し、各行の外接矩形を返す。
/// 入力の前景は0、背景は255。列優先など、行が連続しない配列も受け付ける。
pub fn detect_lines_projection(binary: &Array2<u8>) -> Vec<BBox> {
    let (h, w) = (binary.nrows(), binary.ncols());
    if h == 0 || w == 0 {
        return Vec::new();
    }

    // 行の採否にはしきい値以上かだけが必要なので、到達した時点で計数を終える。
    let min_fg_count = (w as f32 * 0.01).max(1.0) as u32;
    let projection = binary.rows().into_iter().map(|row| match row.as_slice() {
        Some(pixels) => has_min_foreground_simd(pixels, min_fg_count),
        None => {
            row.iter()
                .filter(|&&px| px == 0)
                .take(min_fg_count as usize)
                .count() as u32
                >= min_fg_count
        }
    });

    // 前景のある行の連なりをまとめる。小画像でも文字行は2px以上とする。
    let min_line_height = (h / 50).max(2);
    let mut lines = Vec::new();
    let mut line_start = None;

    // 最後に背景行を1つ補い、画像の下端まで続く文字行も同じ分岐で確定する。
    for (y, has_foreground) in projection.chain(std::iter::once(false)).enumerate() {
        if has_foreground {
            line_start.get_or_insert(y);
        } else if let Some(start) = line_start.take() {
            let line_height = y - start;
            if line_height >= min_line_height {
                lines.push(BBox::new(0, start as u32, w as u32, line_height as u32));
            }
        }
    }

    lines
}

/// 16画素ずつ比較し、必要な前景画素数に達したら残りの走査を省く。
fn has_min_foreground_simd(row: &[u8], minimum: u32) -> bool {
    if minimum == 0 {
        return true;
    }
    let zero_vec = u8x16::splat(0);
    let (chunks, remainder) = row.as_chunks::<16>();
    let mut count: u32 = 0;

    for &chunk in chunks {
        let mask = u8x16::new(chunk).simd_eq(zero_vec).to_bitmask();
        // 背景だけのブロックではビット数の計算も省く。
        if mask != 0 {
            count += mask.count_ones();
            if count >= minimum {
                return true;
            }
        }
    }

    // SIMD幅に満たない末尾だけを個別に数える。
    for &pixel in remainder {
        count += u32::from(pixel == 0);
        if count >= minimum {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, ShapeBuilder, s};

    #[test]
    fn column_major_images_preserve_line_boundaries() {
        let mut img = Array2::from_elem((8, 33).f(), 255u8);
        for y in [0, 1, 5, 6, 7] {
            for x in 1..32 {
                img[[y, x]] = 0;
            }
        }
        assert!(!img.is_standard_layout());
        let boxes: Vec<_> = detect_lines_projection(&img)
            .iter()
            .map(|b| (b.x, b.y, b.width, b.height))
            .collect();
        assert_eq!(boxes, [(0, 0, 33, 2), (0, 5, 33, 3)]);
    }

    #[test]
    fn foreground_threshold_matches_scalar_at_every_simd_boundary() {
        for len in 0..=65 {
            for zero_at in 0..=len {
                let row: Vec<_> = (0..len)
                    .map(|i| if i == zero_at || i % 5 == 0 { 0 } else { 128 })
                    .collect();
                let count = row.iter().filter(|&&px| px == 0).count() as u32;
                for minimum in [0, 1, 2, 15, 16, 17, 32, 64, u32::MAX] {
                    assert_eq!(
                        has_min_foreground_simd(&row, minimum),
                        count >= minimum,
                        "len={len}, minimum={minimum}"
                    );
                }
            }
        }
    }

    #[test]
    fn strided_and_reversed_images_match_contiguous_pixels() {
        for width in [1, 15, 16, 17, 31, 32, 33] {
            let image = Array2::from_shape_fn((12, width).f(), |(y, x)| {
                if !(3..9).contains(&y) && x % 3 == 0 {
                    0u8
                } else {
                    128
                }
            });
            for image in [image.clone(), image.slice_move(s![..;-1, ..;-1])] {
                let standard = Array2::from_shape_fn(image.dim(), |at| image[at]);
                let boxes = |image: &Array2<u8>| {
                    detect_lines_projection(image)
                        .iter()
                        .map(|b| (b.x, b.y, b.width, b.height))
                        .collect::<Vec<_>>()
                };
                assert_eq!(boxes(&image), boxes(&standard), "width={width}");
            }
        }
    }

    #[test]
    fn line_height_and_foreground_thresholds_preserve_trailing_lines() {
        let mut image = Array2::from_elem((8, 200), 255u8);
        image[[0, 0]] = 0;
        image[[1, 0]] = 0;
        for y in [3, 6, 7] {
            image[[y, 0]] = 0;
            image[[y, 1]] = 0;
        }
        let lines = detect_lines_projection(&image);
        assert_eq!(lines.len(), 1);
        assert_eq!((lines[0].y, lines[0].height), (6, 2));
        for shape in [(0, 0), (0, 16), (16, 0)] {
            assert!(detect_lines_projection(&Array2::zeros(shape)).is_empty());
        }
    }

    #[test]
    fn test_detect_lines_empty_image() {
        let img = Array2::from_elem((100, 200), 255u8);
        let lines = detect_lines_projection(&img);
        assert!(lines.is_empty());
    }

    #[test]
    fn test_detect_two_lines() {
        let mut img = Array2::from_elem((100, 200), 255u8);

        // 横書きの文字行を2本描く。
        for y in 10..30 {
            for x in 10..190 {
                img[[y, x]] = 0;
            }
        }
        for y in 60..80 {
            for x in 10..190 {
                img[[y, x]] = 0;
            }
        }

        let lines = detect_lines_projection(&img);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].y < lines[1].y);
    }

    #[test]
    fn test_detect_lines_small_image() {
        // 20x50の小画像に文字行を1本配置する。
        let mut img = Array2::from_elem((20, 50), 255u8);
        for y in 5..15 {
            for x in 2..48 {
                img[[y, x]] = 0;
            }
        }
        let lines = detect_lines_projection(&img);
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn test_detect_lines_large_image() {
        // 4000x6000の大画像に文字行を3本配置する。
        let mut img = Array2::from_elem((4000, 6000), 255u8);
        for y in 200..400 {
            for x in 100..5900 {
                img[[y, x]] = 0;
            }
        }
        for y in 1500..1700 {
            for x in 100..5900 {
                img[[y, x]] = 0;
            }
        }
        for y in 3000..3200 {
            for x in 100..5900 {
                img[[y, x]] = 0;
            }
        }
        let lines = detect_lines_projection(&img);
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_foreground_threshold_accuracy() {
        // 32画素のうち、先頭16画素だけを前景にする。
        let mut row = vec![255u8; 32];
        for item in row.iter_mut().take(16) {
            *item = 0;
        }
        assert!(has_min_foreground_simd(&row, 16));
        assert!(!has_min_foreground_simd(&row, 17));

        // 全画素が前景の場合。
        let all_zero = vec![0u8; 50];
        assert!(has_min_foreground_simd(&all_zero, 50));
        assert!(!has_min_foreground_simd(&all_zero, 51));

        // 前景がない場合。
        let no_zero = vec![128u8; 50];
        assert!(!has_min_foreground_simd(&no_zero, 1));
    }
}
