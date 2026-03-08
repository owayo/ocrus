use crate::charset::Charset;

/// CTC Greedy decode: take argmax at each timestep and collapse repeats, removing blank.
/// `logits` shape: (timesteps, num_classes)
pub fn ctc_greedy_decode(
    logits: &[f32],
    timesteps: usize,
    num_classes: usize,
    charset: &Charset,
) -> (String, f32) {
    if timesteps == 0 || num_classes == 0 {
        return (String::new(), 0.0);
    }

    let mut indices = Vec::with_capacity(timesteps);
    let mut total_confidence = 0.0f32;
    let mut count = 0u32;

    for t in 0..timesteps {
        let offset = t * num_classes;
        let slice = &logits[offset..offset + num_classes];

        let (best_idx, &best_val) = slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        total_confidence += best_val;
        count += 1;

        indices.push(best_idx);
    }

    // Collapse repeats and remove blank
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

/// CTC Greedy decode with logit masking.
/// Same as `ctc_greedy_decode` but applies a boolean mask to suppress disallowed classes.
/// `mask` length must equal `num_classes`. Classes where `mask[i] == false` are set to -inf.
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

        // Apply mask: masked classes get -inf
        let masked: Vec<f32> = slice
            .iter()
            .enumerate()
            .map(|(i, &v)| if mask[i] { v } else { f32::NEG_INFINITY })
            .collect();

        let (best_idx, &best_val) = masked
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        total_confidence += best_val;
        count += 1;

        indices.push(best_idx);
    }

    // Collapse repeats and remove blank
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charset::Charset;

    #[test]
    fn test_ctc_greedy_simple() {
        // 3 timesteps, 4 classes (blank=0, a=1, b=2, c=3)
        let charset = Charset::from_chars(&['a', 'b', 'c']);

        // t0: class 1 (a), t1: class 1 (a), t2: class 2 (b)
        #[rustfmt::skip]
        let logits = vec![
            -10.0,  10.0, -10.0, -10.0, // t0: 'a'
            -10.0,  10.0, -10.0, -10.0, // t1: 'a' (repeat)
            -10.0, -10.0,  10.0, -10.0, // t2: 'b'
        ];

        let (text, _conf) = ctc_greedy_decode(&logits, 3, 4, &charset);
        assert_eq!(text, "ab");
    }

    #[test]
    fn test_ctc_greedy_blank_separation() {
        let charset = Charset::from_chars(&['a', 'b']);

        // t0: 'a', t1: blank, t2: 'a' → "aa"
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
        // 3 classes: blank=0, a=1, b=2
        let charset = Charset::from_chars(&['a', 'b']);

        // Without mask, best would be 'b' at t0, but mask disallows 'b'
        #[rustfmt::skip]
        let logits = vec![
            -10.0, 5.0, 10.0, // t0: 'b' is best, but masked out
        ];

        // mask: blank=true, a=true, b=false
        let mask = vec![true, true, false];
        let (text, _) = ctc_greedy_decode_masked(&logits, 1, 3, &charset, &mask);
        assert_eq!(text, "a"); // forced to pick 'a'

        // Without mask, should pick 'b'
        let (text2, _) = ctc_greedy_decode(&logits, 1, 3, &charset);
        assert_eq!(text2, "b");
    }
}
