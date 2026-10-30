use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum OcrMode {
    /// Recognize each line using its original normalized image.
    #[default]
    Auto,
    /// Vote over the original, thickened and thinned images of each line.
    Accurate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrusConfig {
    pub mode: OcrMode,
    pub model_dir: PathBuf,
    pub num_threads: usize,
}

impl Default for OcrusConfig {
    fn default() -> Self {
        let model_dir = dirs_model_path(|name| std::env::var(name).ok());
        Self {
            mode: OcrMode::Auto,
            model_dir,
            num_threads: num_cpus(),
        }
    }
}

fn dirs_model_path(get_env: impl Fn(&str) -> Option<String>) -> PathBuf {
    if let Some(dir) = get_env("OCRUS_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    let home = get_env("HOME")
        .or_else(|| get_env("USERPROFILE"))
        .unwrap_or_else(|| ".".to_string());
    PathBuf::from(home).join(".ocrus").join("models")
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_deserialization_accepts_only_auto_and_accurate() {
        use serde::de::value::{Error, StrDeserializer};

        let parse = |name| OcrMode::deserialize(StrDeserializer::<Error>::new(name));
        assert!(matches!(OcrMode::default(), OcrMode::Auto));
        assert!(matches!(parse("Auto"), Ok(OcrMode::Auto)));
        assert!(matches!(parse("Accurate"), Ok(OcrMode::Accurate)));
        assert!(parse("Fastest").is_err());
    }

    #[test]
    fn model_directory_environment_precedence() {
        let cases = [
            (vec![], "./.ocrus/models"),
            (vec![("USERPROFILE", "profile")], "profile/.ocrus/models"),
            (
                vec![("HOME", "home"), ("USERPROFILE", "profile")],
                "home/.ocrus/models",
            ),
            (
                vec![
                    ("OCRUS_MODEL_DIR", "custom-models"),
                    ("HOME", "home"),
                    ("USERPROFILE", "profile"),
                ],
                "custom-models",
            ),
        ];
        for (vars, expected) in cases {
            let actual = dirs_model_path(|name| {
                vars.iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| value.to_string())
            });
            assert_eq!(actual, PathBuf::from(expected), "environment: {vars:?}");
        }
    }
}
