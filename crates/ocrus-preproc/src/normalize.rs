use ndarray::Array4;
use ocrus_core::BBox;
use wide::f32x8;

const TARGET_HEIGHT: u32 = 48;
const NUM_CHANNELS: usize = 3;
/// リサイズ後の最大幅。極端に長い行によるメモリ消費を抑える。
const MAX_WIDTH: usize = 2048;
/// モデルのストライドに合わせた、幅の8ピクセル単位の整列。
/// PaddleOCR はバッチ内の最大幅へ余白を加える。単画像では8の倍数に切り上げる。
const WIDTH_ALIGN: usize = 8;
/// PaddleOCR と同じ余白値。ゼロの画素を正規化すると -1.0 になる。
const PAD_VALUE: f32 = -1.0;

/// 正規化の SIMD 定数: (px/255.0 - 0.5)/0.5 = px/127.5 - 1.0
const SIMD_SCALE: f32 = 1.0 / 127.5;
const SIMD_OFFSET: f32 = -1.0;

/// 矩形で行画像を切り出し、固定の高さへリサイズして NCHW の f32 テンソルにする。
/// 出力は (1, 3, TARGET_HEIGHT, new_width)。PaddleOCR の規約で正規化し、
/// 小さい切り出しには余白を加え、長い行は MAX_WIDTH に収める。
pub fn normalize_line(gray: &ndarray::Array2<u8>, bbox: &BBox) -> Array4<f32> {
    normalize_line_scaled(gray, bbox, 1.0)
}

/// 幅の倍率を指定して正規化する。`width_scale > 1.0` は画像本体の幅を保ち、
/// 右側の余白を増やす。1 以下や NaN は本体の幅を保ち、最大幅を超えない。
/// 余白は CTC の時刻数を増やすために使う。
pub fn normalize_line_scaled(
    gray: &ndarray::Array2<u8>,
    bbox: &BBox,
    width_scale: f32,
) -> Array4<f32> {
    normalize_line_inner(gray, bbox, false, width_scale)
}

/// 縦書きの列を切り出し、90度回転して横書きと同じ方法でリサイズする。
pub fn normalize_line_vertical(gray: &ndarray::Array2<u8>, bbox: &BBox) -> Array4<f32> {
    normalize_line_inner(gray, bbox, true, 1.0)
}

fn normalize_line_inner(
    gray: &ndarray::Array2<u8>,
    bbox: &BBox,
    rotate: bool,
    width_scale: f32,
) -> Array4<f32> {
    let (img_h, img_w) = (gray.nrows() as u32, gray.ncols() as u32);

    // 矩形を画像内に収める。座標と寸法の加算でもオーバーフローを防ぐ。
    let x0 = bbox.x.min(img_w) as usize;
    let y0 = bbox.y.min(img_h) as usize;
    let x1 = bbox.x.saturating_add(bbox.width).min(img_w) as usize;
    let y1 = bbox.y.saturating_add(bbox.height).min(img_h) as usize;

    let raw_crop_h = y1 - y0;
    let raw_crop_w = x1 - x0;

    if raw_crop_h == 0 || raw_crop_w == 0 {
        return Array4::from_elem((1, NUM_CHANNELS, TARGET_HEIGHT as usize, 1), PAD_VALUE);
    }

    // 縦書きでは回転後の高さと幅が入れ替わる。
    let (crop_h, crop_w) = if rotate {
        (raw_crop_w, raw_crop_h)
    } else {
        (raw_crop_h, raw_crop_w)
    };

    // 縦横比を保った幅を計算し、MAX_WIDTH を上限とする。
    let scale = TARGET_HEIGHT as f32 / crop_h as f32;
    let content_w = ((crop_w as f32 * scale).round().max(1.0) as usize).min(MAX_WIDTH);
    let new_h = TARGET_HEIGHT as usize;
    // 余白の倍率を適用し、本体の幅以上を確保してストライドの倍数に切り上げる。
    let scaled_w = ((content_w as f32 * width_scale).round() as usize).clamp(content_w, MAX_WIDTH);
    let new_w = scaled_w.div_ceil(WIDTH_ALIGN) * WIDTH_ALIGN;

    // 双線形補間の後に SIMD で正規化する。右側の余白は初期値 PAD_VALUE を保つ。
    let mut resized = Array4::from_elem((1, NUM_CHANNELS, new_h, new_w), PAD_VALUE);

    let scale_vec = f32x8::splat(SIMD_SCALE);
    let offset_vec = f32x8::splat(SIMD_OFFSET);

    // 回転を考慮して切り出し領域の画素を取得し、添字を領域内に収める。
    let sample = |cy: usize, cx: usize| -> u8 {
        if rotate {
            let orig_y = cx.min(raw_crop_h - 1) + y0;
            let orig_x = (raw_crop_w - 1 - cy.min(raw_crop_w - 1)) + x0;
            gray[[orig_y, orig_x]]
        } else {
            gray[[cy.min(crop_h - 1) + y0, cx.min(crop_w - 1) + x0]]
        }
    };

    for ry in 0..new_h {
        let src_y = ry as f32 * crop_h as f32 / new_h as f32;

        // 双線形補間でこの行の画素値を求める。
        let row_pixels: Vec<f32> = (0..content_w)
            .map(|rx| {
                let src_x = rx as f32 * crop_w as f32 / content_w as f32;

                let x0i = (src_x as usize).min(crop_w.saturating_sub(1));
                let y0i = (src_y as usize).min(crop_h.saturating_sub(1));
                let x1i = (x0i + 1).min(crop_w - 1);
                let y1i = (y0i + 1).min(crop_h - 1);

                let xf = src_x - x0i as f32;
                let yf = src_y - y0i as f32;

                let p00 = sample(y0i, x0i) as f32;
                let p10 = sample(y0i, x1i) as f32;
                let p01 = sample(y1i, x0i) as f32;
                let p11 = sample(y1i, x1i) as f32;

                let top = p00 + (p10 - p00) * xf;
                let bot = p01 + (p11 - p01) * xf;
                top + (bot - top) * yf
            })
            .collect();

        // 画像本体を8画素ずつ SIMD で正規化する。
        let chunks = content_w / 8;
        let remainder = content_w % 8;

        for i in 0..chunks {
            let base = i * 8;
            let mut px = [0.0f32; 8];
            px.copy_from_slice(&row_pixels[base..base + 8]);
            let v = f32x8::new(px);
            let normalized = v * scale_vec + offset_vec;
            let arr: [f32; 8] = normalized.to_array();

            for (j, &val) in arr.iter().enumerate() {
                let rx = base + j;
                for c in 0..NUM_CHANNELS {
                    resized[[0, c, ry, rx]] = val;
                }
            }
        }

        // 8画素未満の端数はスカラー演算で処理する。
        let start = chunks * 8;
        for i in 0..remainder {
            let rx = start + i;
            let normalized = row_pixels[rx] * SIMD_SCALE + SIMD_OFFSET;
            for c in 0..NUM_CHANNELS {
                resized[[0, c, ry, rx]] = normalized;
            }
        }
    }

    resized
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ocrus_core::BBox;

    #[test]
    fn test_normalize_line_shape() {
        let gray = Array2::from_elem((100, 200), 128u8);
        let bbox = BBox::new(10, 20, 180, 30);
        let result = normalize_line(&gray, &bbox);
        assert_eq!(result.shape()[0], 1);
        assert_eq!(result.shape()[1], 3);
        assert_eq!(result.shape()[2], 48);
        assert!(result.shape()[3] > 0);
    }

    #[test]
    fn test_normalize_line_values_range() {
        let gray = Array2::from_elem((100, 200), 200u8);
        let bbox = BBox::new(0, 0, 200, 100);
        let result = normalize_line(&gray, &bbox);
        for &v in result.iter() {
            assert!((-1.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_normalize_very_small_image() {
        let gray = Array2::from_elem((5, 10), 100u8);
        let bbox = BBox::new(0, 0, 10, 5);
        let result = normalize_line(&gray, &bbox);
        assert_eq!(result.shape()[0], 1);
        assert_eq!(result.shape()[1], 3);
        assert_eq!(result.shape()[2], 48);
        assert!(result.shape()[3] > 0);
    }

    #[test]
    fn test_normalize_very_wide_line() {
        let gray = Array2::from_elem((50, 10000), 128u8);
        let bbox = BBox::new(0, 0, 10000, 50);
        let result = normalize_line(&gray, &bbox);
        assert_eq!(result.shape()[2], 48);
        assert!(result.shape()[3] <= MAX_WIDTH);
    }

    #[test]
    fn test_normalize_empty_bbox() {
        let gray = Array2::from_elem((100, 200), 128u8);
        let bbox = BBox::new(50, 50, 0, 0);
        let result = normalize_line(&gray, &bbox);
        assert_eq!(result.shape()[2], 48);
        assert_eq!(result.shape()[3], 1);
    }

    #[test]
    fn normalize_bbox_outside_image_returns_padding() {
        let gray = Array2::from_elem((10, 20), 255u8);
        for bbox in [BBox::new(20, 0, 1, 10), BBox::new(0, 10, 20, 1)] {
            for result in [
                normalize_line(&gray, &bbox),
                normalize_line_vertical(&gray, &bbox),
            ] {
                assert_eq!(result.shape(), &[1, 3, 48, 1]);
                assert!(result.iter().all(|&v| v == PAD_VALUE));
            }
        }
    }

    #[test]
    fn normalize_bbox_overflow_is_clipped() {
        let gray = Array2::from_elem((10, 20), 255u8);
        let bbox = BBox::new(1, 1, u32::MAX, u32::MAX);
        let clipped = BBox::new(1, 1, 19, 9);
        assert_eq!(
            normalize_line(&gray, &bbox),
            normalize_line(&gray, &clipped)
        );
        assert_eq!(
            normalize_line_vertical(&gray, &bbox),
            normalize_line_vertical(&gray, &clipped)
        );
    }

    #[test]
    fn normalize_width_scale_preserves_content() {
        let gray = Array2::from_elem((48, 64), 255u8);
        let bbox = BBox::new(0, 0, 64, 48);
        let expected = normalize_line(&gray, &bbox);
        for scale in [0.5, 0.0, -1.0, f32::NAN, f32::NEG_INFINITY] {
            assert_eq!(normalize_line_scaled(&gray, &bbox, scale), expected);
        }
    }

    #[test]
    fn normalize_width_scale_adds_bounded_padding() {
        let gray = Array2::from_elem((48, 13), 255u8);
        let bbox = BBox::new(0, 0, 13, 48);
        for (scale, width) in [(2.0, 32), (f32::INFINITY, MAX_WIDTH)] {
            let result = normalize_line_scaled(&gray, &bbox, scale);
            assert_eq!(result.shape(), &[1, 3, 48, width]);
            assert!(
                result
                    .slice(ndarray::s![.., .., .., 13..])
                    .iter()
                    .all(|&v| v == PAD_VALUE)
            );
        }
    }

    #[test]
    fn normalize_zero_dimension_returns_padding() {
        for shape in [(0, 0), (0, 10), (10, 0)] {
            let gray = Array2::from_elem(shape, 255u8);
            let result = normalize_line(&gray, &BBox::new(0, 0, u32::MAX, u32::MAX));
            assert_eq!(result.shape(), &[1, 3, 48, 1]);
            assert!(result.iter().all(|&v| v == PAD_VALUE));
        }
    }

    #[test]
    fn test_normalize_large_image() {
        let gray = Array2::from_elem((4000, 6000), 128u8);
        let bbox = BBox::new(100, 500, 5800, 200);
        let result = normalize_line(&gray, &bbox);
        assert_eq!(result.shape()[0], 1);
        assert_eq!(result.shape()[1], 3);
        assert_eq!(result.shape()[2], 48);
        assert!(result.shape()[3] <= MAX_WIDTH);
        assert!(result.shape()[3] > 0);
    }

    #[test]
    fn test_normalize_simd_accuracy() {
        // 既知の画素値で SIMD とスカラーの正規化結果を比較する。
        let gray = Array2::from_shape_fn((100, 200), |(y, x)| ((y * 200 + x) % 256) as u8);
        let bbox = BBox::new(0, 0, 200, 100);
        let result = normalize_line(&gray, &bbox);

        for &v in result.iter() {
            assert!(
                (-1.0..=1.001).contains(&v),
                "value {v} out of expected range"
            );
        }
    }

    #[test]
    fn test_normalize_line_vertical_shape() {
        // 幅が狭く、背の高い縦書きの列。
        let gray = Array2::from_elem((500, 200), 128u8);
        let bbox = BBox::new(50, 0, 30, 500);
        let result = normalize_line_vertical(&gray, &bbox);
        assert_eq!(result.shape()[0], 1);
        assert_eq!(result.shape()[1], 3);
        assert_eq!(result.shape()[2], 48); // height is always TARGET_HEIGHT
        // 回転後は元の高さ500が幅になり、出力幅は 500 * 48/30 ≈ 800。
        assert!(result.shape()[3] > 100, "rotated width should be large");
    }

    #[test]
    fn test_normalize_line_vertical_values() {
        let gray = Array2::from_elem((200, 100), 200u8);
        let bbox = BBox::new(10, 0, 20, 200);
        let result = normalize_line_vertical(&gray, &bbox);
        for &v in result.iter() {
            assert!((-1.0..=1.001).contains(&v));
        }
    }
}
