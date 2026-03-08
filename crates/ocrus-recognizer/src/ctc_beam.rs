use std::collections::HashMap;

use crate::charset::Charset;

/// CTC Prefix Beam Search デコード。
/// あいまいな入力で greedy より精度が出る経路を探索し、(text, confidence) を返す。
pub fn ctc_beam_decode(
    logits: &[f32],
    timesteps: usize,
    num_classes: usize,
    charset: &Charset,
    beam_width: usize,
) -> (String, f32) {
    if timesteps == 0 || num_classes == 0 || beam_width == 0 {
        return (String::new(), 0.0);
    }
    let Some(expected_len) = timesteps.checked_mul(num_classes) else {
        return (String::new(), 0.0);
    };
    if logits.len() < expected_len {
        return (String::new(), 0.0);
    }

    // ビーム状態: prefix -> (log_p_blank, log_p_non_blank)
    let neg_inf = f64::NEG_INFINITY;
    let mut beams: HashMap<String, (f64, f64)> = HashMap::new();
    beams.insert(String::new(), (0.0, neg_inf)); // 初期状態: 空prefixのblank確率のみ1.0

    for t in 0..timesteps {
        let offset = t * num_classes;
        let raw = &logits[offset..offset + num_classes];

        // 現在時刻の log-softmax
        let log_probs = log_softmax(raw);

        let mut new_beams: HashMap<String, (f64, f64)> = HashMap::new();

        for (prefix, &(log_pb, log_pnb)) in &beams {
            let log_p_prefix = log_sum_exp(log_pb, log_pnb);

            // 1) blank へ遷移
            let log_p_blank_new = log_p_prefix + log_probs[0];
            let entry = new_beams
                .entry(prefix.clone())
                .or_insert((neg_inf, neg_inf));
            entry.0 = log_sum_exp(entry.0, log_p_blank_new);

            // 2) 各文字へ遷移
            for (c_idx, &log_p_c) in log_probs.iter().enumerate().skip(1) {
                let ch = match charset.index_to_char(c_idx) {
                    Some(ch) => ch,
                    None => continue,
                };

                let last_char = prefix.chars().last();

                if last_char == Some(ch) {
                    // 同一文字:
                    // non-blank 経路は同じ prefix に残る (CTC の重複畳み込み)
                    let log_p_stay = log_pnb + log_p_c;
                    let entry = new_beams
                        .entry(prefix.clone())
                        .or_insert((neg_inf, neg_inf));
                    entry.1 = log_sum_exp(entry.1, log_p_stay);

                    // blank 経路からは同一文字を追加できる (blank で重複が分離)
                    let mut new_prefix = prefix.clone();
                    new_prefix.push(ch);
                    let log_p_ext = log_pb + log_p_c;
                    let entry = new_beams.entry(new_prefix).or_insert((neg_inf, neg_inf));
                    entry.1 = log_sum_exp(entry.1, log_p_ext);
                } else {
                    // 異なる文字は prefix を拡張
                    let mut new_prefix = prefix.clone();
                    new_prefix.push(ch);
                    let log_p_new = log_p_prefix + log_p_c;
                    let entry = new_beams.entry(new_prefix).or_insert((neg_inf, neg_inf));
                    entry.1 = log_sum_exp(entry.1, log_p_new);
                }
            }
        }

        // 上位 beam_width 件だけ残す
        let mut scored: Vec<(String, (f64, f64))> = new_beams.into_iter().collect();
        scored.sort_by(|a, b| {
            let sa = log_sum_exp(a.1.0, a.1.1);
            let sb = log_sum_exp(b.1.0, b.1.1);
            sb.total_cmp(&sa)
        });
        scored.truncate(beam_width);

        beams = scored.into_iter().collect();
    }

    // 最終的な最良ビームを選ぶ
    let (best_text, (log_pb, log_pnb)) = beams
        .into_iter()
        .max_by(|a, b| {
            let sa = log_sum_exp(a.1.0, a.1.1);
            let sb = log_sum_exp(b.1.0, b.1.1);
            sa.total_cmp(&sb)
        })
        .unwrap_or((String::new(), (neg_inf, neg_inf)));

    let log_confidence = log_sum_exp(log_pb, log_pnb);
    let confidence = (log_confidence / timesteps as f64).exp() as f32;

    (best_text, confidence)
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let sanitized: Vec<f64> = logits
        .iter()
        .map(|&x| {
            if x.is_finite() {
                x as f64
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let max = sanitized.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return vec![f64::NEG_INFINITY; logits.len()];
    }
    let sum_exp: f64 = sanitized.iter().map(|&x| (x - max).exp()).sum();
    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return vec![f64::NEG_INFINITY; logits.len()];
    }
    let log_sum = sum_exp.ln();
    sanitized.into_iter().map(|x| x - max - log_sum).collect()
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charset::Charset;

    #[test]
    fn test_beam_simple() {
        // 4クラス: blank=0, a=1, b=2, c=3
        let charset = Charset::from_chars(&['a', 'b', 'c']);
        let logits = vec![
            -10.0, 10.0, -10.0, -10.0, // t0: 'a'
            -10.0, 10.0, -10.0, -10.0, // t1: 'a' (重複)
            -10.0, -10.0, 10.0, -10.0, // t2: 'b'
        ];
        let (text, _conf) = ctc_beam_decode(&logits, 3, 4, &charset, 5);
        assert_eq!(text, "ab");
    }

    #[test]
    fn test_beam_blank_separation() {
        // 3クラス: blank=0, a=1, b=2
        let charset = Charset::from_chars(&['a', 'b']);
        let logits = vec![
            -10.0, 10.0, -10.0, // t0: 'a'
            10.0, -10.0, -10.0, // t1: blank
            -10.0, 10.0, -10.0, // t2: 'a'
        ];
        let (text, _) = ctc_beam_decode(&logits, 3, 3, &charset, 5);
        assert_eq!(text, "aa");
    }

    #[test]
    fn test_beam_ambiguous_prefers_better_path() {
        // 3クラス: blank=0, a=1, b=2
        let charset = Charset::from_chars(&['a', 'b']);
        let logits = vec![
            0.1, -0.1, -10.0, // t0: blank が 'a' よりわずかに高い
            0.1, -0.1, -10.0, // t1: 同条件
            -10.0, 5.0, -10.0, // t2: 明確に 'a'
        ];
        let (text, _) = ctc_beam_decode(&logits, 3, 3, &charset, 5);
        assert!(!text.is_empty(), "Beam search should find non-empty text");
    }

    #[test]
    fn test_beam_width_1_matches_greedy_text() {
        // 3クラス: blank=0, x=1, y=2
        let charset = Charset::from_chars(&['x', 'y']);
        let logits = vec![-10.0, 10.0, -10.0, -10.0, -10.0, 10.0];
        let (text, _) = ctc_beam_decode(&logits, 2, 3, &charset, 1);
        assert_eq!(text, "xy");
    }

    #[test]
    fn test_beam_short_logits_returns_empty() {
        let charset = Charset::from_chars(&['a']);
        let logits = vec![1.0, 0.0, 1.0];
        let (text, conf) = ctc_beam_decode(&logits, 2, 2, &charset, 5);
        assert_eq!(text, "");
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_beam_nan_does_not_panic() {
        let charset = Charset::from_chars(&['a']);
        let logits = vec![f32::NAN, 1.0];
        let (text, conf) = ctc_beam_decode(&logits, 1, 2, &charset, 5);
        assert_eq!(text, "a");
        assert!(conf.is_finite());
    }
}
