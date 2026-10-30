//! CTC デコーダが共通して行うロジット列の検証。

/// ロジット列を `(timesteps, num_classes)` として扱えるか検証し、使用する範囲を返す。
///
/// 時刻数かクラス数が 0 の場合、その積が `usize` で表せない場合、
/// 与えられたロジットが必要な長さに足りない場合は `None` を返す。
/// 呼び出し元はデコード結果を空として扱う。
///
/// 戻り値は必要な長さに切り詰めてあるため、`chunks_exact(num_classes)` で
/// 時刻ごとに走査できる。
pub(crate) fn validated_frames(
    logits: &[f32],
    timesteps: usize,
    num_classes: usize,
) -> Option<&[f32]> {
    if timesteps == 0 || num_classes == 0 {
        return None;
    }
    let expected_len = timesteps.checked_mul(num_classes)?;
    logits.get(..expected_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_trimmed_to_the_expected_length() {
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0];
        let frames = validated_frames(&logits, 2, 2).unwrap();
        assert_eq!(frames, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(frames.len() % 2, 0);
    }

    #[test]
    fn empty_dimensions_and_short_input_are_rejected() {
        let logits = [1.0, 2.0, 3.0];
        assert!(validated_frames(&logits, 0, 2).is_none());
        assert!(validated_frames(&logits, 2, 0).is_none());
        assert!(validated_frames(&logits, 2, 2).is_none());
    }

    #[test]
    fn an_overflowing_product_is_rejected() {
        let logits = [1.0];
        assert!(validated_frames(&logits, usize::MAX, 2).is_none());
    }
}
