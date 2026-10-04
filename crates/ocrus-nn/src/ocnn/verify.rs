//! Checking that a model file computes what its converter measured.
//!
//! A model can load perfectly and still be wrong: this project lost an afternoon to an
//! `.ocnn` built by an older converter, which answered with confidence 0.9 and was wrong
//! on every character. Structural validation cannot catch that — only running the thing
//! and comparing against a recorded result can.
//!
//! The converter records, for a handful of widths, the argmax sequence produced by the
//! reference runtime on a pseudo-random input. The input is regenerated from a seed, so
//! each record costs a few hundred bytes rather than a megabyte.

use ocrus_core::error::Result;

use crate::ocnn::exec::Executor;
use crate::tensor::NdTensor;

/// Outcome of checking every golden record in a model.
#[derive(Debug, Clone, Default)]
pub struct VerifyReport {
    pub checked: usize,
    pub failures: Vec<String>,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Deterministic pseudo-random input, identical to the converter's generator.
///
/// A plain LCG: the values only need to be reproducible and to exercise the whole graph,
/// not to be statistically good.
pub fn golden_input(seed: u32, width: usize) -> NdTensor<f32> {
    let mut state = seed;
    let mut data = Vec::with_capacity(3 * 48 * width);
    for _ in 0..3 * 48 * width {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = (state >> 8) as f64 / 16_777_216.0;
        data.push((unit * 2.0 - 1.0) as f32);
    }
    NdTensor::from_vec(data, &[1, 3, 48, width])
}

/// Class index with the highest score at each timestep.
pub fn argmax_sequence(output: &NdTensor<f32>) -> Vec<u32> {
    if output.ndim() != 3 {
        return Vec::new();
    }
    let (timesteps, classes) = (output.shape[1], output.shape[2]);
    (0..timesteps)
        .map(|t| {
            let row = &output.data[t * classes..(t + 1) * classes];
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        })
        .collect()
}

/// Run every golden record and report the ones that disagree.
///
/// Returns Ok with an empty failure list when the model reproduces what was recorded.
/// A model with no golden records reports `checked == 0`, which callers should treat as
/// "unverifiable" rather than "verified".
pub fn verify(exec: &Executor) -> Result<VerifyReport> {
    let mut report = VerifyReport::default();

    for (i, golden) in exec.model().meta.golden.iter().enumerate() {
        let input = golden_input(golden.seed, golden.width as usize);
        let output = exec.run(input)?;
        report.checked += 1;

        if output.shape != golden.out_shape {
            report.failures.push(format!(
                "golden {i} (W={}): output shape {:?}, expected {:?}",
                golden.width, output.shape, golden.out_shape
            ));
            continue;
        }

        let got = argmax_sequence(&output);
        if got != golden.argmax {
            let first = got
                .iter()
                .zip(&golden.argmax)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            report.failures.push(format!(
                "golden {i} (W={}): argmax differs at timestep {first} \
                 (got {}, expected {}); {} of {} timesteps disagree",
                golden.width,
                got.get(first).copied().unwrap_or(0),
                golden.argmax.get(first).copied().unwrap_or(0),
                got.iter()
                    .zip(&golden.argmax)
                    .filter(|(a, b)| a != b)
                    .count(),
                got.len(),
            ));
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_input_is_deterministic_and_bounded() {
        let a = golden_input(7, 16);
        let b = golden_input(7, 16);
        assert_eq!(a.data, b.data);
        assert_eq!(a.shape, vec![1, 3, 48, 16]);
        assert!(a.data.iter().all(|v| (-1.0..=1.0).contains(v)));
        assert_ne!(golden_input(8, 16).data, a.data);
    }

    #[test]
    fn argmax_picks_the_largest_class_per_timestep() {
        let t = NdTensor::from_vec(vec![0.1, 0.9, 0.3, 0.8, 0.2, 0.1], &[1, 2, 3]);
        assert_eq!(argmax_sequence(&t), vec![1, 0]);
    }
}
