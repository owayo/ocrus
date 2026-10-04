use crate::charset::Charset;

/// 単文字を中心とする入力を、時刻ごとの確率を集計してデコードする。
///
/// blank 以外の各クラスへの支持を合算し、最大のクラスを選ぶ。
/// 信頼度はパディングを除いた時刻の平均確率として返す。
pub fn ctc_tla_decode(
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

    let mut class_votes = vec![0.0f64; num_classes];
    let mut informative_frames = 0usize;

    for t in 0..timesteps {
        let offset = t * num_classes;
        let slice = &logits[offset..offset + num_classes];

        let max_val = slice
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        if !max_val.is_finite() {
            continue;
        }

        let exps: Vec<f64> = slice
            .iter()
            .map(|&x| {
                if x.is_finite() {
                    ((x - max_val) as f64).exp()
                } else {
                    0.0
                }
            })
            .collect();
        let sum_exp: f64 = exps.iter().sum();
        if !sum_exp.is_finite() || sum_exp <= 0.0 {
            continue;
        }

        // blank の確率が極めて高い時刻はパディングとして集計から外す。
        let blank_prob = exps[charset.blank_index()] / sum_exp;
        if blank_prob > 0.95 {
            continue;
        }
        informative_frames += 1;

        for (idx, &exp_val) in exps.iter().enumerate().skip(1) {
            class_votes[idx] += exp_val / sum_exp;
        }
    }

    let mut best_idx = 0usize;
    let mut best_vote = 0.0f64;
    for (idx, &vote) in class_votes.iter().enumerate().skip(1) {
        if vote > best_vote {
            best_idx = idx;
            best_vote = vote;
        }
    }

    let Some(ch) = charset.index_to_char(best_idx) else {
        return (String::new(), 0.0);
    };

    let confidence = (best_vote / informative_frames as f64).clamp(0.0, 1.0) as f32;
    (ch.to_string(), confidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctc_greedy::ctc_greedy_decode;

    #[test]
    fn tla_confidence_does_not_grow_with_frame_count() {
        let charset = Charset::from_chars(&['a']);
        let (_, single) = ctc_tla_decode(&[-10.0, 5.0], 1, 2, &charset);
        let (_, repeated) = ctc_tla_decode(&[-10.0, 5.0, -10.0, 5.0], 2, 2, &charset);
        assert!((0.0..=1.0).contains(&repeated));
        assert_eq!(single, repeated);
    }

    #[test]
    fn test_tla_prefers_consistent_character_over_blank() {
        let charset = Charset::from_chars(&['a', 'b']);
        let logits = vec![
            4.0, 3.9, -10.0, // blank が最大だが 'a' への支持も近い
            4.0, 3.9, -10.0, //
            -10.0, 5.0, -10.0, // 'a' が明確に最大
        ];

        let (greedy_text, _) = ctc_greedy_decode(&logits, 3, 3, &charset);
        let (tla_text, tla_conf) = ctc_tla_decode(&logits, 3, 3, &charset);

        assert_eq!(greedy_text, "a");
        assert_eq!(tla_text, "a");
        assert!(tla_conf > 0.0);
    }

    #[test]
    fn test_tla_skips_blank_dominated_padding() {
        let charset = Charset::from_chars(&['a']);
        let logits = vec![
            -10.0, 5.0, // 文字への支持がある時刻
            12.0, -12.0, // パディング
            12.0, -12.0, // パディング
        ];

        let (text, _) = ctc_tla_decode(&logits, 3, 2, &charset);
        assert_eq!(text, "a");
    }

    #[test]
    fn test_tla_short_logits_returns_empty() {
        let charset = Charset::from_chars(&['a']);
        let (text, conf) = ctc_tla_decode(&[1.0], 1, 2, &charset);
        assert_eq!(text, "");
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_tla_nan_does_not_panic() {
        let charset = Charset::from_chars(&['a']);
        let logits = vec![f32::NAN, 1.0];
        let (text, conf) = ctc_tla_decode(&logits, 1, 2, &charset);
        assert_eq!(text, "a");
        assert!(conf.is_finite());
    }
}
