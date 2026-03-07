use std::path::Path;
use std::sync::{Arc, Mutex};

use ndarray::ArrayD;
use ort::session::Session;
use ort::value::TensorRef;

use ocrus_core::OcrusError;
use ocrus_core::error::Result;

use crate::backend::{
    InferenceBackend, MAX_BATCH_SIZE, ModelHandle, ModelOptions, Quantization, Tensor,
};

/// Padding value for width-padding in batched inference (normalized white).
const PAD_VALUE: f32 = 1.0;

/// ONNX Runtime inference backend.
pub struct OrtBackend;

impl OrtBackend {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }
}

impl Default for OrtBackend {
    fn default() -> Self {
        Self
    }
}

impl InferenceBackend for OrtBackend {
    fn load_model(&self, path: &Path, opts: &ModelOptions) -> Result<ModelHandle> {
        let mut builder = Session::builder()
            .map_err(|e| OcrusError::Runtime(format!("Failed to create session builder: {e}")))?;

        builder = builder
            .with_intra_threads(opts.num_threads)
            .map_err(|e| OcrusError::Runtime(format!("Failed to set threads: {e}")))?;

        if opts.quantization != Quantization::None {
            // INT8 quantization is performed offline via the Python script
            // `scripts/ocrus_scripts/quantize.py` (onnxruntime.quantization.quantize_dynamic).
            // At runtime we simply load the pre-quantized ONNX model and apply
            // Level3 graph optimizations for best performance.
            builder = builder
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                .map_err(|e| {
                    OcrusError::Runtime(format!("Failed to set optimization level: {e}"))
                })?;
        }

        let session = builder.commit_from_file(path).map_err(|e| {
            OcrusError::Model(format!("Failed to load model {}: {e}", path.display()))
        })?;

        Ok(ModelHandle {
            inner: Box::new(Arc::new(Mutex::new(session))),
        })
    }

    fn run(&self, handle: &ModelHandle, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let session_mutex = handle
            .inner
            .downcast_ref::<Arc<Mutex<Session>>>()
            .ok_or_else(|| OcrusError::Runtime("Invalid model handle".to_string()))?;

        let mut session = session_mutex
            .lock()
            .map_err(|e| OcrusError::Runtime(format!("Failed to lock session: {e}")))?;

        // Build ndarray inputs and TensorRef views
        let arrays: Vec<ArrayD<f32>> = inputs
            .iter()
            .map(|tensor| {
                ArrayD::from_shape_vec(ndarray::IxDyn(&tensor.shape), tensor.data.clone())
                    .map_err(|e| OcrusError::Runtime(format!("Tensor shape mismatch: {e}")))
            })
            .collect::<Result<_>>()?;

        let tensor_refs: Vec<TensorRef<'_, f32>> = arrays
            .iter()
            .map(|arr| {
                TensorRef::from_array_view(arr.view())
                    .map_err(|e| OcrusError::Runtime(format!("Failed to create tensor ref: {e}")))
            })
            .collect::<Result<_>>()?;

        let input_values: Vec<ort::session::SessionInputValue<'_>> =
            tensor_refs.into_iter().map(|t| t.into()).collect();

        let outputs = session
            .run(input_values.as_slice())
            .map_err(|e| OcrusError::Runtime(format!("Inference failed: {e}")))?;

        let mut result = Vec::new();
        for (_name, output) in outputs.iter() {
            let (shape_info, data_slice) = output
                .try_extract_tensor::<f32>()
                .map_err(|e| OcrusError::Runtime(format!("Failed to extract output: {e}")))?;
            let shape: Vec<usize> = shape_info.iter().map(|&d| d as usize).collect();
            let data = data_slice.to_vec();
            result.push(Tensor::new(data, shape));
        }

        Ok(result)
    }

    fn run_batch(&self, handle: &ModelHandle, batch: &[Tensor]) -> Result<Vec<Vec<Tensor>>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        if batch.len() == 1 {
            return Ok(vec![self.run(handle, &[batch[0].clone()])?]);
        }

        let mut all_results = Vec::with_capacity(batch.len());

        for chunk in batch.chunks(MAX_BATCH_SIZE) {
            let chunk_results = self.run_batch_chunk(handle, chunk)?;
            all_results.extend(chunk_results);
        }

        Ok(all_results)
    }
}

impl OrtBackend {
    /// Run a single batch chunk (up to MAX_BATCH_SIZE tensors).
    /// Each input tensor is expected to have shape (1, C, H, W_i).
    /// They are padded to (N, C, H, max_W) for batched inference.
    fn run_batch_chunk(&self, handle: &ModelHandle, chunk: &[Tensor]) -> Result<Vec<Vec<Tensor>>> {
        let n = chunk.len();

        // Validate and extract dimensions from first tensor
        if chunk[0].shape.len() != 4 || chunk[0].shape[0] != 1 {
            // Fall back to individual runs for non-standard shapes
            return chunk
                .iter()
                .map(|t| self.run(handle, std::slice::from_ref(t)))
                .collect();
        }

        let channels = chunk[0].shape[1];
        let height = chunk[0].shape[2];

        // Find max width across all tensors
        let max_width = chunk
            .iter()
            .map(|t| if t.shape.len() == 4 { t.shape[3] } else { 0 })
            .max()
            .unwrap_or(0);

        // Build padded batch tensor (N, C, H, max_W)
        let batch_size = n * channels * height * max_width;
        let mut batch_data = vec![PAD_VALUE; batch_size];

        for (i, tensor) in chunk.iter().enumerate() {
            let w = tensor.shape[3];
            // Copy each pixel, padding is already PAD_VALUE
            for c in 0..channels {
                for h in 0..height {
                    let src_offset = c * height * w + h * w;
                    let dst_offset =
                        i * channels * height * max_width + c * height * max_width + h * max_width;
                    batch_data[dst_offset..dst_offset + w]
                        .copy_from_slice(&tensor.data[src_offset..src_offset + w]);
                }
            }
        }

        let batch_tensor = Tensor::new(batch_data, vec![n, channels, height, max_width]);

        // Run batched inference
        let outputs = self.run(handle, &[batch_tensor])?;

        // Split output (N, T, C_out) into N individual (1, T, C_out) tensors
        let mut results = Vec::with_capacity(n);
        for output in &outputs {
            if output.shape.len() == 3 && output.shape[0] == n {
                let t_dim = output.shape[1];
                let c_dim = output.shape[2];
                let stride = t_dim * c_dim;

                for i in 0..n {
                    let start = i * stride;
                    let end = start + stride;
                    let individual =
                        Tensor::new(output.data[start..end].to_vec(), vec![1, t_dim, c_dim]);
                    if i >= results.len() {
                        results.push(vec![individual]);
                    } else {
                        results[i].push(individual);
                    }
                }
            } else {
                // Non-standard output shape: return as-is for each item
                for i in 0..n {
                    if i >= results.len() {
                        results.push(vec![output.clone()]);
                    } else {
                        results[i].push(output.clone());
                    }
                }
            }
        }

        Ok(results)
    }
}
