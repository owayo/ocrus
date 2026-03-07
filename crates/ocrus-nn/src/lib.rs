pub mod arena;
pub mod graph;
pub mod model;
pub mod ops;
pub mod tensor;

use std::path::Path;

use ocrus_core::error::Result;

/// Inference tensor (compatible with existing pipeline)
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

/// Neural network inference engine (replaces OrtBackend)
pub struct NnEngine;

impl NnEngine {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    /// Load a .ocnn model
    pub fn load_model(&self, path: &Path) -> Result<model::OcnnModel> {
        model::OcnnModel::load(path)
    }

    /// Run inference on a single input
    pub fn run(&self, model_data: &model::OcnnModel, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let input = &inputs[0];
        let nd_input = tensor::NdTensor::from_vec(input.data.clone(), &input.shape);
        let mut ar = arena::TensorArena::new();

        let output = graph::execute(model_data, nd_input, &mut ar)?;

        Ok(vec![Tensor {
            data: output.data,
            shape: output.shape,
        }])
    }

    /// Run batched inference (pad to max width, run as single batch)
    pub fn run_batch(
        &self,
        model_data: &model::OcnnModel,
        batch: &[Tensor],
    ) -> Result<Vec<Vec<Tensor>>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        if batch.len() == 1 {
            return Ok(vec![self.run(model_data, &[batch[0].clone()])?]);
        }

        // Validate shapes: expect (1, C, H, W_i) for each tensor
        if batch[0].shape.len() != 4 || batch[0].shape[0] != 1 {
            // Fall back to individual runs
            return batch
                .iter()
                .map(|t| self.run(model_data, std::slice::from_ref(t)))
                .collect();
        }

        let n = batch.len();
        let channels = batch[0].shape[1];
        let height = batch[0].shape[2];
        let max_width = batch.iter().map(|t| t.shape[3]).max().unwrap_or(0);

        // Build padded batch tensor (N, C, H, max_W)
        let batch_size = n * channels * height * max_width;
        let mut batch_data = vec![1.0f32; batch_size]; // PAD_VALUE = 1.0

        for (i, t) in batch.iter().enumerate() {
            let w = t.shape[3];
            for c in 0..channels {
                for h in 0..height {
                    let src_offset = c * height * w + h * w;
                    let dst_offset =
                        i * channels * height * max_width + c * height * max_width + h * max_width;
                    batch_data[dst_offset..dst_offset + w]
                        .copy_from_slice(&t.data[src_offset..src_offset + w]);
                }
            }
        }

        let batch_tensor = Tensor::new(batch_data, vec![n, channels, height, max_width]);
        let outputs = self.run(model_data, &[batch_tensor])?;

        // Split output (N, T, C_out) into individual (1, T, C_out)
        let mut results = Vec::with_capacity(n);
        if let Some(output) = outputs.first() {
            if output.shape.len() == 3 && output.shape[0] == n {
                let t_dim = output.shape[1];
                let c_dim = output.shape[2];
                let stride = t_dim * c_dim;
                for i in 0..n {
                    let start = i * stride;
                    let end = start + stride;
                    results.push(vec![Tensor::new(
                        output.data[start..end].to_vec(),
                        vec![1, t_dim, c_dim],
                    )]);
                }
            } else {
                for _ in 0..n {
                    results.push(vec![output.clone()]);
                }
            }
        }

        Ok(results)
    }
}

impl Default for NnEngine {
    fn default() -> Self {
        Self
    }
}

#[cfg(test)]
mod engine_tests {
    use super::*;
    use memmap2::MmapMut;
    use model::{LayerDescriptor, LayerType, build_ocnn};

    fn make_model(layers: &[(LayerDescriptor, &[u8])]) -> model::OcnnModel {
        let data = build_ocnn(layers);
        let mut mm = MmapMut::map_anon(data.len()).unwrap();
        mm.copy_from_slice(&data);
        let mmap = mm.make_read_only().unwrap();
        model::OcnnModel::from_mmap(mmap).unwrap()
    }

    #[test]
    fn test_nn_engine_run() {
        let engine = NnEngine::new().unwrap();

        // Model: ReLU only
        let relu = LayerDescriptor {
            layer_type: LayerType::ReLU,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
        };
        let m = make_model(&[(relu, &[])]);

        let input = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![1, 4]);
        let outputs = engine.run(&m, &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].data, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_nn_engine_empty_input() {
        let engine = NnEngine::new().unwrap();
        let relu = LayerDescriptor {
            layer_type: LayerType::ReLU,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
        };
        let m = make_model(&[(relu, &[])]);
        let outputs = engine.run(&m, &[]).unwrap();
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_nn_engine_default() {
        let _engine = NnEngine::default();
    }
}
