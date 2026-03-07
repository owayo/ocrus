use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum OcrMode {
    #[default]
    Auto,
    Fastest,
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
        let model_dir = dirs_model_path();
        Self {
            mode: OcrMode::Auto,
            model_dir,
            num_threads: num_cpus(),
        }
    }
}

fn dirs_model_path() -> PathBuf {
    if let Ok(dir) = std::env::var("OCRUS_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".ocrus").join("models")
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
