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
    ctc_beam_decode_topk(
        logits,
        timesteps,
        num_classes,
        charset,
        beam_width,
        num_classes.saturating_sub(1),
    )
}

/// CTC Prefix Beam Search with per-timestep Top-K pruning.
///
/// `top_k` limits the number of non-blank classes expanded at each timestep.
/// Blank is always retained so repeated-character paths remain reachable.
pub fn ctc_beam_decode_topk(
    logits: &[f32],
    timesteps: usize,
    num_classes: usize,
    charset: &Charset,
    beam_width: usize,
    top_k: usize,
) -> (String, f32) {
    if timesteps == 0 || num_classes == 0 || beam_width == 0 {
        return (String::new(), 0.0);
    }
    let Some(expected_len) = timesteps.checked_mul(num_classes) else {
        return (String::new(), 0.0);
    };
    if logits.len() < expected_len || top_k == 0 {
        return (String::new(), 0.0);
    }

    // ビーム状態: prefix -> (log_p_blank, log_p_non_blank)
    let neg_inf = f64::NEG_INFINITY;
    let mut beams: HashMap<String, (f64, f64)> = HashMap::new();
    beams.insert(String::new(), (0.0, neg_inf)); // 初期状態: 空prefixのblank確率のみ1.0

    for t in 0..timesteps {
        let offset = t * num_classes;
        let raw = &logits[offset..offset + num_classes];

        let log_softmax = LogSoftmax::new(raw);
        let top_candidates = top_k_indices(raw, top_k);

        let mut new_beams: HashMap<String, (f64, f64)> = HashMap::new();

        for (prefix, &(log_pb, log_pnb)) in &beams {
            let log_p_prefix = log_sum_exp(log_pb, log_pnb);

            // 1) blank へ遷移
            let log_p_blank = log_softmax.at(0);
            if log_p_blank.is_finite() {
                let log_p_blank_new = log_p_prefix + log_p_blank;
                let entry = new_beams
                    .entry(prefix.clone())
                    .or_insert((neg_inf, neg_inf));
                entry.0 = log_sum_exp(entry.0, log_p_blank_new);
            }

            // 2) 各文字へ遷移
            for c_idx in top_candidates.iter().copied() {
                let log_p_c = log_softmax.at(c_idx);
                if !log_p_c.is_finite() {
                    continue;
                }
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

struct LogSoftmax<'a> {
    logits: &'a [f32],
    max: f64,
    log_sum: Option<f64>,
}

impl<'a> LogSoftmax<'a> {
    fn new(logits: &'a [f32]) -> Self {
        let max = logits
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .map(f64::from)
            .fold(f64::NEG_INFINITY, f64::max);
        let log_sum = if max.is_finite() {
            let sum_exp: f64 = logits
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .map(|x| (f64::from(x) - max).exp())
                .sum();
            if sum_exp.is_finite() && sum_exp > 0.0 {
                Some(sum_exp.ln())
            } else {
                None
            }
        } else {
            None
        };

        Self {
            logits,
            max,
            log_sum,
        }
    }

    fn at(&self, idx: usize) -> f64 {
        let Some(log_sum) = self.log_sum else {
            return f64::NEG_INFINITY;
        };
        let Some(&value) = self.logits.get(idx) else {
            return f64::NEG_INFINITY;
        };
        if !value.is_finite() {
            return f64::NEG_INFINITY;
        }
        f64::from(value) - self.max - log_sum
    }
}

fn top_k_indices(logits: &[f32], top_k: usize) -> Vec<usize> {
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .skip(1)
        .filter(|(_, value)| value.is_finite())
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    candidates.truncate(top_k.min(candidates.len()));
    candidates.into_iter().map(|(idx, _)| idx).collect()
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
    use crate::ctc_greedy::ctc_greedy_decode;

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

    #[test]
    fn test_beam_topk_matches_full_when_target_in_topk() {
        let charset = Charset::from_chars(&['a', 'b', 'c']);
        let logits = vec![
            -10.0, 9.0, 1.0, 0.0, //
            -10.0, 9.0, 1.0, 0.0, //
            -10.0, 0.0, 8.0, 1.0, //
        ];

        let (full_text, _) = ctc_beam_decode(&logits, 3, 4, &charset, 5);
        let (topk_text, _) = ctc_beam_decode_topk(&logits, 3, 4, &charset, 5, 2);

        assert_eq!(full_text, "ab");
        assert_eq!(topk_text, full_text);
    }

    #[test]
    fn test_beam_topk_width_one_matches_greedy_for_single_best_path() {
        let charset = Charset::from_chars(&['x', 'y']);
        let logits = vec![
            -10.0, 10.0, -10.0, //
            -10.0, -10.0, 10.0, //
        ];

        let (greedy_text, _) = ctc_greedy_decode(&logits, 2, 3, &charset);
        let (beam_text, _) = ctc_beam_decode_topk(&logits, 2, 3, &charset, 1, 1);

        assert_eq!(beam_text, greedy_text);
    }

    #[test]
    fn test_beam_topk_zero_returns_empty() {
        let charset = Charset::from_chars(&['a']);
        let logits = vec![0.0, 1.0];
        let (text, conf) = ctc_beam_decode_topk(&logits, 1, 2, &charset, 5, 0);
        assert_eq!(text, "");
        assert_eq!(conf, 0.0);
    }
}
