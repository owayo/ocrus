//! Recover missing word separators from image gaps and the decoded CTC positions.
use ocrus_core::BBox;
use ocrus_nn::NdTensor;
use ocrus_recognizer::ctc_greedy::CharacterSpan;

pub(super) fn is_word_character(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

fn can_separate(left: char, right: char) -> bool {
    // Dots, colons and quotes also occur inside URLs, decimals and identifiers.
    // Their CTC peaks can overlap a neighboring glyph; leave those gaps to the model.
    let word_end = is_word_character(left) || ",!?)]}|-+<>=".contains(left);
    let word_start = is_word_character(right) || matches!(right, '-' | '+' | '|' | '<' | '>' | '=');
    word_end && word_start
}

// A missed underscore or narrow glyph can look like a word gap. Preserve complete
// path/URL/decimal-like runs rather than guessing boundaries inside their components.
fn identifier_boundaries(spans: &[CharacterSpan]) -> Vec<bool> {
    let mut protected = vec![false; spans.len()];
    let mut start = 0;
    while start < spans.len() {
        let mut end = start;
        while end < spans.len()
            && (is_word_character(spans[end].character)
                || "_./@:?&=%~-".contains(spans[end].character))
        {
            end += 1;
        }
        if end > start {
            if spans[start..end]
                .iter()
                .any(|span| "_./@".contains(span.character))
            {
                protected[start + 1..end].fill(true);
            }
            start = end;
        } else {
            start += 1;
        }
    }
    protected
}

pub(super) fn restore_word_spaces(
    text: &str,
    spans: &[CharacterSpan],
    input: &NdTensor<f32>,
    timesteps: usize,
    bbox: &BBox,
) -> String {
    if spans.len() < 2 || timesteps == 0 || input.shape.len() != 4 {
        return text.to_owned();
    }
    let (h, w) = (input.shape[2], input.shape[3]);
    if h == 0 || w == 0 || input.data.len() < h * w || bbox.width == 0 || bbox.height == 0 {
        return text.to_owned();
    }
    let centers: Vec<f32> = spans
        .iter()
        .map(|s| (s.start + s.end) as f32 * w as f32 / (2.0 * timesteps as f32))
        .collect();
    let mut distances: Vec<f32> = spans
        .windows(2)
        .zip(centers.windows(2))
        .filter(|(s, _)| is_word_character(s[0].character) && is_word_character(s[1].character))
        .map(|(_, x)| x[1] - x[0])
        .filter(|&d| d > 0.0)
        .collect();
    if distances.is_empty() {
        return text.to_owned();
    }
    distances.sort_by(f32::total_cmp);
    let compression = (w as f32 * bbox.height as f32 / (bbox.width as f32 * h as f32)).min(1.0);
    let pitch = distances[distances.len() / 2].min(h as f32 * 0.9 * compression);
    if pitch < 4.0 {
        return text.to_owned();
    }
    // Normalization duplicates the grayscale plane into all three channels.
    let channel = &input.data[..h * w];
    let light_on_dark = channel.iter().filter(|&&v| v < 0.0).count() > h * w / 2;
    let occupied: Vec<bool> = (0..w)
        .map(|x| {
            (0..h)
                .filter(|&y| {
                    let value = channel[y * w + x];
                    if light_on_dark {
                        value > 0.0
                    } else {
                        value < 0.0
                    }
                })
                .take(2)
                .count()
                >= 2
        })
        .collect();
    let mut gaps = Vec::new();
    let mut seen_ink = false;
    let mut run = 0usize;
    for &ink in &occupied {
        if ink {
            if seen_ink && run > 0 {
                gaps.push(run);
            }
            seen_ink = true;
            run = 0;
        } else if seen_ink {
            run += 1;
        }
    }
    gaps.sort_unstable();
    let usual_gap = gaps.get(gaps.len() / 2).copied().unwrap_or(0) as f32;
    // Reject normal glyph side bearings: require a gap wider than both the
    // typical inter-glyph gap and a fraction of the normalized letter height.
    let minimum_gap = (pitch * 0.55)
        .max(usual_gap.min(pitch * 0.75) * 1.4)
        .max(h as f32 * compression * 0.3);
    let protected = identifier_boundaries(spans);
    let mut result = String::with_capacity(text.len() + spans.len() / 4);
    for (index, span) in spans.iter().enumerate() {
        if index > 0 {
            let previous = spans[index - 1].character;
            let current = span.character;
            if !protected[index]
                && can_separate(previous, current)
                && centers[index] - centers[index - 1] >= pitch * 1.5
            {
                let left = centers[index - 1].ceil().max(0.0) as usize;
                let right = (centers[index].floor().max(0.0) as usize).min(w);
                let mut longest = 0;
                let mut run = 0;
                for &ink in occupied.get(left..right).unwrap_or_default() {
                    if ink {
                        run = 0;
                    } else {
                        run += 1;
                        longest = longest.max(run);
                    }
                }
                let radius = (pitch * 0.5).ceil() as usize;
                let has_ink_near = |center: f32| {
                    let x = center.max(0.0) as usize;
                    occupied
                        .get(x.saturating_sub(radius)..x.saturating_add(radius).min(w))
                        .is_some_and(|pixels| pixels.iter().any(|&ink| ink))
                };
                let boundary_gap = if is_word_character(previous) && is_word_character(current) {
                    minimum_gap
                } else {
                    minimum_gap.max(pitch * 0.85)
                };
                if longest as f32 >= boundary_gap
                    && has_ink_near(centers[index - 1])
                    && has_ink_near(centers[index])
                {
                    result.push(' ');
                }
            }
        }
        result.push(span.character);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image_and_spans(
        text: &str,
        thin: bool,
    ) -> (String, Vec<CharacterSpan>, NdTensor<f32>, BBox) {
        let width = (text.chars().count() * 40 + 40).div_ceil(8) * 8;
        let mut input = NdTensor::from_vec(vec![1.; 3 * 48 * width], &[1, 3, 48, width]);
        let mut spans = Vec::new();
        let mut plain = String::new();
        for (cell, c) in text.chars().enumerate() {
            if c == ' ' {
                continue;
            }
            let center = 20 + cell * 40;
            spans.push(CharacterSpan {
                character: c,
                start: center / 8,
                end: center / 8 + 1,
            });
            plain.push(c);
            for y in 8..40 {
                for x in center - if thin { 4 } else { 12 }..center + if thin { 4 } else { 12 } {
                    input.data[y * width + x] = -1.;
                }
            }
        }
        (plain, spans, input, BBox::new(0, 0, width as u32, 48))
    }
    #[test]
    fn terminal_summary_recovers_word_spaces_without_changing_characters() {
        let text = "4 files changed, 48 insertions(+), 4 deletions(-)";
        let (plain, spans, input, bbox) = image_and_spans(text, false);
        let restored = restore_word_spaces(&plain, &spans, &input, input.shape[3] / 8, &bbox);
        assert_eq!(restored, text);
        assert_eq!(restored.replace(' ', ""), plain);
    }
    #[test]
    fn narrow_glyphs_code_and_urls_do_not_gain_spaces() {
        for text in [
            "iiiiWWWWmmmmllll",
            "https://example.com/path?a=1&b=2",
            "console.log(helloWorld)",
            "foo_bar/baz.txt",
            "日本語の認識",
        ] {
            let (plain, spans, input, bbox) = image_and_spans(text, true);
            assert_eq!(
                restore_word_spaces(&plain, &spans, &input, input.shape[3] / 8, &bbox),
                plain
            );
        }
    }
    #[test]
    fn gaps_inside_path_decimal_and_url_runs_are_not_recovered() {
        for text in [
            "foo/ bar",
            "rep ort_2026.txt",
            "hello@example. com",
            "~/Documents/samp le",
            "v2.4. 1",
            "3. 14",
        ] {
            let (plain, spans, input, bbox) = image_and_spans(text, false);
            assert_eq!(
                restore_word_spaces(&plain, &spans, &input, input.shape[3] / 8, &bbox),
                plain,
            );
        }
    }
    #[test]
    fn fullwidth_characters_do_not_use_the_ascii_pitch() {
        assert!(!can_separate('１', '２'));
        assert!(!can_separate('１', 'a'));
        assert!(!can_separate('a', 'Ａ'));
        assert!(can_separate('a', '2'));
    }
    #[test]
    fn existing_space_tokens_are_kept() {
        let (plain, mut spans, input, bbox) = image_and_spans("hello world", false);
        spans.insert(
            5,
            CharacterSpan {
                character: ' ',
                start: 27,
                end: 28,
            },
        );
        let text = plain.chars().take(5).collect::<String>()
            + " "
            + &plain.chars().skip(5).collect::<String>();
        assert_eq!(
            restore_word_spaces(&text, &spans, &input, input.shape[3] / 8, &bbox),
            "hello world"
        );
    }
    #[test]
    fn omitted_narrow_glyphs_and_blank_images_do_not_create_boundaries() {
        let (mut plain, mut spans, mut input, bbox) = image_and_spans("iiii", true);
        spans.remove(1);
        plain.remove(1);
        assert_eq!(
            restore_word_spaces(&plain, &spans, &input, input.shape[3] / 8, &bbox),
            plain
        );
        input.data.fill(1.0);
        assert_eq!(
            restore_word_spaces(&plain, &spans, &input, input.shape[3] / 8, &bbox),
            plain
        );
    }
    #[test]
    fn faint_punctuation_and_shifted_ctc_peaks_do_not_split_identifiers() {
        for (text, shift) in [("foo.bar()", 4), ("foo(bar)", 4), ("foo.bar()", -4)] {
            let (plain, mut spans, mut input, bbox) = image_and_spans(text, true);
            let width = input.shape[3];
            for y in 0..48 {
                for x in 136..144 {
                    input.data[y * width + x] = 1.0;
                }
            }
            input.data[24 * width + 140] = -1.0;
            spans[3].start = spans[3].start.checked_add_signed(shift).unwrap();
            spans[3].end = spans[3].end.checked_add_signed(shift).unwrap();
            assert_eq!(
                restore_word_spaces(&plain, &spans, &input, width / 8, &bbox),
                plain
            );
        }
    }
}
