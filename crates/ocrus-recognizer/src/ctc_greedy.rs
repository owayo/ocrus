use crate::charset::Charset;

/// CTC Greedy デコード: 各時刻で argmax を取り、繰り返しと blank を畳み込む。
/// `logits` 形状: (timesteps, num_classes)
pub fn ctc_greedy_decode(
    logits: &[f32],
    timesteps: usize,
    num_classes: usize,
    charset: &Charset,
) -> (String, f32) {
    if timesteps == 0 || num_classes == 0 {
        return (String::new(), 0.0);
    }
    let Some(expected_len) = timesteps.checked_mul(num_classes) else {
        return (String::new(), 0.0);
    };
    if logits.len() < expected_len {
        return (String::new(), 0.0);
    }

    let mut indices = Vec::with_capacity(timesteps);
    let mut total_confidence = 0.0f32;
    let mut count = 0u32;

    for t in 0..timesteps {
        let offset = t * num_classes;
        let slice = &logits[offset..offset + num_classes];
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &v) in slice.iter().enumerate() {
            let candidate = sanitize_logit(v);
            if candidate.total_cmp(&best_val).is_gt() {
                best_idx = idx;
                best_val = candidate;
            }
        }
        if best_val.is_finite() {
            total_confidence += best_val;
            count += 1;
        }

        indices.push(best_idx);
    }

    // 連続重複を畳み込み、blank を除去
    let mut result = String::new();
    let mut prev = None;

    for &idx in &indices {
        if Some(idx) == prev {
            continue;
        }
        prev = Some(idx);
        if idx != charset.blank_index()
            && let Some(ch) = charset.index_to_char(idx)
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
    if timesteps == 0 || num_classes == 0 {
        return (String::new(), 0.0);
    }
    let Some(expected_len) = timesteps.checked_mul(num_classes) else {
        return (String::new(), 0.0);
    };
    if logits.len() < expected_len {
        return (String::new(), 0.0);
    }

    assert_eq!(
        mask.len(),
        num_classes,
        "mask length must equal num_classes"
    );

    let mut indices = Vec::with_capacity(timesteps);
    let mut total_confidence = 0.0f32;
    let mut count = 0u32;

    for t in 0..timesteps {
        let offset = t * num_classes;
        let slice = &logits[offset..offset + num_classes];
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &v) in slice.iter().enumerate() {
            let candidate = if mask[idx] {
                sanitize_logit(v)
            } else {
                f32::NEG_INFINITY
            };
            if candidate.total_cmp(&best_val).is_gt() {
                best_idx = idx;
                best_val = candidate;
            }
        }
        if best_val.is_finite() {
            total_confidence += best_val;
            count += 1;
        }

        indices.push(best_idx);
    }

    // 連続重複を畳み込み、blank を除去
    let mut result = String::new();
    let mut prev = None;

    for &idx in &indices {
        if Some(idx) == prev {
            continue;
        }
        prev = Some(idx);
        if idx != charset.blank_index()
            && let Some(ch) = charset.index_to_char(idx)
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

fn sanitize_logit(v: f32) -> f32 {
    if v.is_finite() { v } else { f32::NEG_INFINITY }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charset::Charset;

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
    fn test_ctc_greedy_masked_nan_does_not_panic() {
        let charset = Charset::from_chars(&['a']);
        let logits = vec![f32::NAN, 1.0];
        let mask = vec![true, true];
        let (text, conf) = ctc_greedy_decode_masked(&logits, 1, 2, &charset, &mask);
        assert_eq!(text, "a");
        assert_eq!(conf, 1.0);
    }
}
