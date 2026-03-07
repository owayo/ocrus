use std::path::PathBuf;

use crate::OcrusConfig;

/// Character set restriction
#[derive(Debug, Clone, Copy, Default)]
pub enum CharsetMode {
    #[default]
    Full,
    Jis,
}

/// Configuration for OCR engine, built via builder pattern
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_dir: PathBuf,
    pub num_threads: usize,
    pub mode: crate::OcrMode,
    pub charset: CharsetMode,
    pub dict_path: Option<PathBuf>,
    pub beam_width: usize,
    pub confidence_threshold: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let defaults = OcrusConfig::default();
        Self {
            model_dir: defaults.model_dir,
            num_threads: 1,
            mode: crate::OcrMode::Auto,
            charset: CharsetMode::Full,
            dict_path: None,
            beam_width: 5,
            confidence_threshold: 0.5,
        }
    }
}

/// Builder for EngineConfig
pub struct EngineConfigBuilder {
    config: EngineConfig,
}

impl EngineConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: EngineConfig::default(),
        }
    }

    pub fn model_dir(mut self, path: PathBuf) -> Self {
        self.config.model_dir = path;
        self
    }

    pub fn num_threads(mut self, n: usize) -> Self {
        self.config.num_threads = n;
        self
    }

    pub fn mode(mut self, mode: crate::OcrMode) -> Self {
        self.config.mode = mode;
        self
    }

    pub fn charset(mut self, charset: CharsetMode) -> Self {
        self.config.charset = charset;
        self
    }

    pub fn dict_path(mut self, path: PathBuf) -> Self {
        self.config.dict_path = Some(path);
        self
    }

    pub fn beam_width(mut self, width: usize) -> Self {
        self.config.beam_width = width;
        self
    }

    pub fn confidence_threshold(mut self, threshold: f32) -> Self {
        self.config.confidence_threshold = threshold;
        self
    }

    pub fn build(self) -> EngineConfig {
        self.config
    }
}

impl Default for EngineConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
