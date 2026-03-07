use std::path::Path;

use ocrus_core::error::Result;

/// Maximum batch size for batched inference.
pub const MAX_BATCH_SIZE: usize = 32;

/// Quantization mode for model loading.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Quantization {
    #[default]
    None,
    Int8,
    Fp16,
}

/// Opaque handle to a loaded model.
#[derive(Debug)]
pub struct ModelHandle {
    pub(crate) inner: Box<dyn std::any::Any + Send + Sync>,
}

/// Options for loading a model.
#[derive(Debug, Clone)]
pub struct ModelOptions {
    pub num_threads: usize,
    pub quantization: Quantization,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            num_threads: 1,
            quantization: Quantization::None,
        }
    }
}

/// Multi-dimensional tensor for inference I/O.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

/// Abstraction over inference backends (ONNX Runtime, etc.).
pub trait InferenceBackend: Send + Sync {
    fn load_model(&self, path: &Path, opts: &ModelOptions) -> Result<ModelHandle>;
    fn run(&self, handle: &ModelHandle, inputs: &[Tensor]) -> Result<Vec<Tensor>>;

    /// Run batched inference. Default implementation runs each tensor individually.
    fn run_batch(&self, handle: &ModelHandle, batch: &[Tensor]) -> Result<Vec<Vec<Tensor>>> {
        batch
            .iter()
            .map(|t| self.run(handle, std::slice::from_ref(t)))
            .collect()
    }
}
