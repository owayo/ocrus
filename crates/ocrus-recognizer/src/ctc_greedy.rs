use crate::charset::Charset;
use crate::ctc::validated_frames;

/// CTC Greedy デコード: 各時刻で argmax を取り、繰り返しと blank を畳み込む。
/// `logits` 形状: (timesteps, num_classes)
pub fn ctc_greedy_decode(
    logits: &[f32],
    timesteps: usize,
    num_classes: usize,
    charset: &Charset,
) -> (String, f32) {
    let Some(frames) = validated_frames(logits, timesteps, num_classes) else {
        return (String::new(), 0.0);
    };

    decode_argmax(frames, num_classes, charset, |_idx, v| sanitize_logit(v))
}

/// 非許可クラスをマスクする CTC Greedy デコード。
/// `mask` の長さは `num_classes` と一致する必要がある。
/// `mask[i] == false` のクラスは `-inf` として扱う。
pub fn ctc_greedy_decode_masked(
    logits: &[f32],
    timesteps: usize,
    num_classes: usize,
    charset: &Charset,
    mask: &[bool],
) -> (String, f32) {
    let Some(frames) = validated_frames(logits, timesteps, num_classes) else {
        return (String::new(), 0.0);
    };

    assert_eq!(
        mask.len(),
        num_classes,
        "mask length must equal num_classes"
    );

    decode_argmax(frames, num_classes, charset, |idx, v| {
        if mask[idx] {
            sanitize_logit(v)
        } else {
            f32::NEG_INFINITY
        }
    })
}

/// 時刻ごとの argmax を取り、文字列と平均信頼度を組み立てる。
///
/// `score` は比較に使う値へロジットを変換する。候補から外すクラスには
/// `f32::NEG_INFINITY` を返す。マスクの有無はこの関数の呼び出し側が決めるため、
/// 検証・畳み込み・信頼度の計算は 1 か所で済む。
///
/// 連続重複の畳み込みと blank の除去は argmax と同じ走査で行うので、
/// 時刻ごとの添字を別に保持しない。
#[inline]
fn decode_argmax(
    frames: &[f32],
    num_classes: usize,
    charset: &Charset,
    score: impl Fn(usize, f32) -> f32,
) -> (String, f32) {
    let blank = charset.blank_index();
    let mut result = String::new();
    let mut total_confidence = 0.0f32;
    let mut count = 0u32;
    let mut prev = None;

    for frame in frames.chunks_exact(num_classes) {
        let (best_idx, best_val) = frame_argmax(frame, &score);
        if best_val.is_finite() {
            total_confidence += best_val;
            count += 1;
        }

        if Some(best_idx) == prev {
            continue;
        }
        prev = Some(best_idx);
        if best_idx != blank
            && let Some(ch) = charset.index_to_char(best_idx)
        {
            result.push(ch);
        }
    }

    let avg_confidence = if count > 0 {
        total_confidence / count as f32
    } else {
        0.0
    };

    (result, avg_confidence)
}

/// Time interval occupied by one collapsed CTC token (end is exclusive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharacterSpan {
    pub character: char,
    pub start: usize,
    pub end: usize,
}

/// Character positions using the same finite-score, tie and mask rules as greedy decoding.
pub fn ctc_greedy_spans(
    logits: &[f32],
    timesteps: usize,
    num_classes: usize,
    charset: &Charset,
    mask: Option<&[bool]>,
) -> Vec<CharacterSpan> {
    let Some(frames) = validated_frames(logits, timesteps, num_classes) else {
        return Vec::new();
    };
    if let Some(mask) = mask {
        assert_eq!(
            mask.len(),
            num_classes,
            "mask length must equal num_classes"
        );
    }
    let mut spans: Vec<CharacterSpan> = Vec::new();
    let mut previous = None;
    for (time, frame) in frames.chunks_exact(num_classes).enumerate() {
        let (index, _) = frame_argmax(frame, &|index, value| {
            if mask.is_none_or(|m| m[index]) {
                sanitize_logit(value)
            } else {
                f32::NEG_INFINITY
            }
        });
        if previous == Some(index) {
            if index != charset.blank_index()
                && charset.index_to_char(index).is_some()
                && let Some(span) = spans.last_mut()
            {
                span.end = time + 1;
            }
        } else if index != charset.blank_index()
            && let Some(character) = charset.index_to_char(index)
        {
            spans.push(CharacterSpan {
                character,
                start: time,
                end: time + 1,
            });
        }
        previous = Some(index);
    }
    spans
}

#[inline]
fn frame_argmax(frame: &[f32], score: &impl Fn(usize, f32) -> f32) -> (usize, f32) {
    let mut best = (0, f32::NEG_INFINITY);
    for (index, &value) in frame.iter().enumerate() {
        let candidate = score(index, value);
        if candidate.total_cmp(&best.1).is_gt() {
            best = (index, candidate);
        }
    }
    best
}

fn sanitize_logit(v: f32) -> f32 {
    if v.is_finite() { v } else { f32::NEG_INFINITY }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charset::Charset;

    #[test]
    fn spans_preserve_collapsed_tokens_spaces_and_masking() {
        let charset = Charset::from_chars(&['a', 'b', ' ']);
        let indices = [1, 1, 0, 1, 3, 3, 2, 2];
        let logits: Vec<f32> = indices
            .iter()
            .flat_map(|&token| (0..4).map(move |i| if i == token { 1.0 } else { -1.0 }))
            .collect();
        let spans = ctc_greedy_spans(&logits, 8, 4, &charset, None);
        assert_eq!(
            spans.iter().map(|s| s.character).collect::<String>(),
            ctc_greedy_decode(&logits, 8, 4, &charset).0
        );
        assert_eq!(
            spans.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(),
            [(0, 2), (3, 4), (4, 6), (6, 8)]
        );
        let mask = [true, true, true, false];
        let spans = ctc_greedy_spans(&logits, 8, 4, &charset, Some(&mask));
        assert_eq!(
            spans.iter().map(|s| s.character).collect::<String>(),
            ctc_greedy_decode_masked(&logits, 8, 4, &charset, &mask).0
        );
    }

    #[test]
    fn unknown_classes_do_not_extend_the_previous_character() {
        let charset = Charset::from_chars(&['a']);
        let spans = ctc_greedy_spans(&[0., 1., 0., 0., 0., 1., 0., 0., 1.], 3, 3, &charset, None);
        assert_eq!(
            spans,
            [CharacterSpan {
                character: 'a',
                start: 0,
                end: 1
            }]
        );
        assert!(ctc_greedy_spans(&[f32::NAN; 4], 2, 2, &charset, None).is_empty());
        assert!(ctc_greedy_spans(&[], 1, 2, &charset, None).is_empty());
    }

    #[test]
    fn test_ctc_greedy_simple() {
        // 3時刻, 4クラス (blank=0, a=1, b=2, c=3)
        let charset = Charset::from_chars(&['a', 'b', 'c']);

        // t0: class 1 (a), t1: class 1 (a), t2: class 2 (b) を想定
        #[rustfmt::skip]
        let logits = vec![
            -10.0,  10.0, -10.0, -10.0, // t0: 'a'
            -10.0,  10.0, -10.0, -10.0, // t1: 'a' (重複)
            -10.0, -10.0,  10.0, -10.0, // t2: 'b'
        ];

        let (text, _conf) = ctc_greedy_decode(&logits, 3, 4, &charset);
        assert_eq!(text, "ab");
    }

    #[test]
    fn test_ctc_greedy_blank_separation() {
        let charset = Charset::from_chars(&['a', 'b']);

        // t0: 'a', t1: blank, t2: 'a' -> "aa"
        #[rustfmt::skip]
        let logits = vec![
            -10.0,  10.0, -10.0, // t0: 'a'
            10.0, -10.0, -10.0, // t1: blank
            -10.0,  10.0, -10.0, // t2: 'a'
        ];

        let (text, _) = ctc_greedy_decode(&logits, 3, 3, &charset);
        assert_eq!(text, "aa");
    }

    #[test]
    fn test_ctc_greedy_masked() {
        // 3クラス: blank=0, a=1, b=2
        let charset = Charset::from_chars(&['a', 'b']);

        // マスクなしなら t0 は 'b' だが、マスクで 'b' を禁止
        #[rustfmt::skip]
        let logits = vec![
            -10.0, 5.0, 10.0, // t0: 'b' is best, but masked out
        ];

        // mask: blank=true, a=true, b=false
        let mask = vec![true, true, false];
        let (text, _) = ctc_greedy_decode_masked(&logits, 1, 3, &charset, &mask);
        assert_eq!(text, "a"); // 強制的に 'a' を選ぶ

        // マスクなしでは 'b' を選ぶ
        let (text2, _) = ctc_greedy_decode(&logits, 1, 3, &charset);
        assert_eq!(text2, "b");
    }

    #[test]
    fn test_ctc_greedy_short_logits_returns_empty() {
        let charset = Charset::from_chars(&['a']);
        let logits = vec![1.0, 0.0, 1.0];
        let (text, conf) = ctc_greedy_decode(&logits, 2, 2, &charset);
        assert_eq!(text, "");
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_ctc_greedy_nan_does_not_panic() {
        let charset = Charset::from_chars(&['a']);
        let logits = vec![f32::NAN, 1.0];
        let (text, conf) = ctc_greedy_decode(&logits, 1, 2, &charset);
        assert_eq!(text, "a");
        assert_eq!(conf, 1.0);
    }

    #[test]
    #[should_panic(expected = "mask length must equal num_classes")]
    fn masked_decode_rejects_a_mask_of_the_wrong_length() {
        let charset = Charset::from_chars(&['a']);
        let mask = vec![true];
        let _ = ctc_greedy_decode_masked(&[1.0, 2.0], 1, 2, &charset, &mask);
    }

    #[test]
    fn a_short_input_is_rejected_before_the_mask_is_checked() {
        // 入力の検証で打ち切る経路では mask の長さを見ない。マスクなし版と
        // 同じ順序で検証することの回帰テスト。
        let charset = Charset::from_chars(&['a']);
        let mask = vec![true];
        let (text, conf) = ctc_greedy_decode_masked(&[1.0], 2, 2, &charset, &mask);
        assert_eq!(text, "");
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_ctc_greedy_masked_nan_does_not_panic() {
        let charset = Charset::from_chars(&['a']);
        let logits = vec![f32::NAN, 1.0];
        let mask = vec![true, true];
        let (text, conf) = ctc_greedy_decode_masked(&logits, 1, 2, &charset, &mask);
        assert_eq!(text, "a");
        assert_eq!(conf, 1.0);
    }
}
