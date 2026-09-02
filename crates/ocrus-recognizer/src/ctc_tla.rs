use crate::charset::Charset;

/// Temporal Logit Aggregation decode for single-character-heavy inputs.
///
/// Unlike greedy CTC, this aggregates per-timestep probabilities for each
/// non-blank class and returns the class with the strongest total support.
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

        // Heavily blank-dominated frames are usually padding.
        let blank_prob = exps[charset.blank_index()] / sum_exp;
        if blank_prob > 0.95 {
            continue;
        }

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

    (ch.to_string(), best_vote as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctc_greedy::ctc_greedy_decode;

    #[test]
    fn test_tla_prefers_consistent_character_over_blank() {
        let charset = Charset::from_chars(&['a', 'b']);
        let logits = vec![
            4.0, 3.9, -10.0, // mostly blank but 'a' is close
            4.0, 3.9, -10.0, //
            -10.0, 5.0, -10.0, // clear 'a'
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
            -10.0, 5.0, // informative
            12.0, -12.0, // padding
            12.0, -12.0, // padding
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
