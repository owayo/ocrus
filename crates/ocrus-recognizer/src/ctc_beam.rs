use std::collections::HashMap;

use crate::charset::Charset;

/// CTC prefix beam search decode.
/// Returns (text, confidence) with better accuracy than greedy for ambiguous inputs.
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

    // beams: prefix -> (log_p_blank, log_p_non_blank)
    let neg_inf = f64::NEG_INFINITY;
    let mut beams: HashMap<String, (f64, f64)> = HashMap::new();
    beams.insert(String::new(), (0.0, neg_inf)); // empty prefix starts with p_blank=1 (log=0)

    for t in 0..timesteps {
        let offset = t * num_classes;
        let raw = &logits[offset..offset + num_classes];

        // Log-softmax for this timestep
        let log_probs = log_softmax(raw);

        let mut new_beams: HashMap<String, (f64, f64)> = HashMap::new();

        for (prefix, &(log_pb, log_pnb)) in &beams {
            let log_p_prefix = log_sum_exp(log_pb, log_pnb);

            // Case 1: extend with blank
            let log_p_blank_new = log_p_prefix + log_probs[0];
            let entry = new_beams
                .entry(prefix.clone())
                .or_insert((neg_inf, neg_inf));
            entry.0 = log_sum_exp(entry.0, log_p_blank_new);

            // Case 2: extend with each character
            for (c_idx, &log_p_c) in log_probs.iter().enumerate().skip(1) {
                let ch = match charset.index_to_char(c_idx) {
                    Some(ch) => ch,
                    None => continue,
                };

                let last_char = prefix.chars().last();

                if last_char == Some(ch) {
                    // Same character as last:
                    // non-blank path: stays on same prefix (CTC repeat collapse)
                    let log_p_stay = log_pnb + log_p_c;
                    let entry = new_beams
                        .entry(prefix.clone())
                        .or_insert((neg_inf, neg_inf));
                    entry.1 = log_sum_exp(entry.1, log_p_stay);

                    // blank path: extends prefix (blank separated the repeats)
                    let mut new_prefix = prefix.clone();
                    new_prefix.push(ch);
                    let log_p_ext = log_pb + log_p_c;
                    let entry = new_beams.entry(new_prefix).or_insert((neg_inf, neg_inf));
                    entry.1 = log_sum_exp(entry.1, log_p_ext);
                } else {
                    // Different character: extend prefix
                    let mut new_prefix = prefix.clone();
                    new_prefix.push(ch);
                    let log_p_new = log_p_prefix + log_p_c;
                    let entry = new_beams.entry(new_prefix).or_insert((neg_inf, neg_inf));
                    entry.1 = log_sum_exp(entry.1, log_p_new);
                }
            }
        }

        // Prune to beam_width
        let mut scored: Vec<(String, (f64, f64))> = new_beams.into_iter().collect();
        scored.sort_by(|a, b| {
            let sa = log_sum_exp(a.1.0, a.1.1);
            let sb = log_sum_exp(b.1.0, b.1.1);
            sb.partial_cmp(&sa).unwrap()
        });
        scored.truncate(beam_width);

        beams = scored.into_iter().collect();
    }

    // Select best beam
    let (best_text, (log_pb, log_pnb)) = beams
        .into_iter()
        .max_by(|a, b| {
            let sa = log_sum_exp(a.1.0, a.1.1);
            let sb = log_sum_exp(b.1.0, b.1.1);
            sa.partial_cmp(&sb).unwrap()
        })
        .unwrap_or((String::new(), (neg_inf, neg_inf)));

    let log_confidence = log_sum_exp(log_pb, log_pnb);
    let confidence = (log_confidence / timesteps as f64).exp() as f32;

    (best_text, confidence)
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum_exp: f64 = logits.iter().map(|&x| ((x as f64) - max).exp()).sum();
    let log_sum = sum_exp.ln();
    logits.iter().map(|&x| (x as f64) - max - log_sum).collect()
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
        let charset = Charset::from_chars(&['a', 'b', 'c']);
        let logits = vec![
            -10.0, 10.0, -10.0, -10.0, // t0: 'a'
            -10.0, 10.0, -10.0, -10.0, // t1: 'a' (repeat)
            -10.0, -10.0, 10.0, -10.0, // t2: 'b'
        ];
        let (text, _conf) = ctc_beam_decode(&logits, 3, 4, &charset, 5);
        assert_eq!(text, "ab");
    }

    #[test]
    fn test_beam_blank_separation() {
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
        let charset = Charset::from_chars(&['a', 'b']);
        let logits = vec![
            0.1, -0.1, -10.0, // t0: blank slightly better than 'a'
            0.1, -0.1, -10.0, // t1: same
            -10.0, 5.0, -10.0, // t2: clearly 'a'
        ];
        let (text, _) = ctc_beam_decode(&logits, 3, 3, &charset, 5);
        assert!(!text.is_empty(), "Beam search should find non-empty text");
    }

    #[test]
    fn test_beam_width_1_matches_greedy_text() {
        let charset = Charset::from_chars(&['x', 'y']);
        let logits = vec![-10.0, 10.0, -10.0, -10.0, -10.0, 10.0];
        let (text, _) = ctc_beam_decode(&logits, 2, 3, &charset, 1);
        assert_eq!(text, "xy");
    }
}
