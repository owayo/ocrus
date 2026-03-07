use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use ocrus_core::OcrusError;
use ocrus_core::error::Result;

/// Magic bytes for .ocnn format
pub const MAGIC: &[u8; 4] = b"OCNN";
pub const VERSION: u32 = 1;

/// Layer types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LayerType {
    Conv2d = 1,
    ConvDepthwise = 2,
    BatchNorm = 3,
    ReLU = 4,
    HardSwish = 5,
    MaxPool2d = 6,
    AvgPool2d = 7,
    Linear = 8,
    Reshape = 9,
    Flatten = 10,
    Transpose = 11,
}

impl LayerType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Conv2d),
            2 => Some(Self::ConvDepthwise),
            3 => Some(Self::BatchNorm),
            4 => Some(Self::ReLU),
            5 => Some(Self::HardSwish),
            6 => Some(Self::MaxPool2d),
            7 => Some(Self::AvgPool2d),
            8 => Some(Self::Linear),
            9 => Some(Self::Reshape),
            10 => Some(Self::Flatten),
            11 => Some(Self::Transpose),
            _ => None,
        }
    }
}

/// Layer descriptor (fixed size: 64 bytes)
#[derive(Debug, Clone)]
pub struct LayerDescriptor {
    pub layer_type: LayerType,
    pub param_offset: u64,
    pub param_size: u64,
    /// Layer-specific config (e.g., in_ch, out_ch, kh, kw, stride_h, stride_w, pad_h, pad_w, ...)
    pub config: [u32; 10],
}

/// Loaded .ocnn model (mmap-backed)
pub struct OcnnModel {
    _mmap: Mmap,
    pub layers: Vec<LayerDescriptor>,
    pub weights_base: *const u8,
    pub weights_len: usize,
}

// SAFETY: weights_base points into mmap which is immutable and lives as long as OcnnModel
unsafe impl Send for OcnnModel {}
unsafe impl Sync for OcnnModel {}

impl OcnnModel {
    /// Load a .ocnn model file via mmap
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| OcrusError::Model(format!("Failed to open model: {e}")))?;
        // SAFETY: model file is read-only and not modified during inference
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| OcrusError::Model(format!("Failed to mmap model: {e}")))?;

        Self::from_mmap(mmap)
    }

    /// Parse an OcnnModel from an already-mapped region.
    /// Also used by tests to avoid writing temp files.
    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        if mmap.len() < 12 {
            return Err(OcrusError::Model("Model file too small".into()));
        }

        // Parse header
        if &mmap[0..4] != MAGIC {
            return Err(OcrusError::Model("Invalid magic bytes".into()));
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(OcrusError::Model(format!("Unsupported version: {version}")));
        }
        let num_layers = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;

        // Parse layer descriptors (each 64 bytes, starting at offset 12)
        let desc_start = 12;
        let desc_size = 64;
        let weights_start = desc_start + num_layers * desc_size;

        if mmap.len() < weights_start {
            return Err(OcrusError::Model("Truncated layer descriptors".into()));
        }

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let offset = desc_start + i * desc_size;
            let buf = &mmap[offset..offset + desc_size];

            let layer_type = LayerType::from_u8(buf[0])
                .ok_or_else(|| OcrusError::Model(format!("Unknown layer type: {}", buf[0])))?;
            // bytes 1..8: padding/reserved
            let param_offset = u64::from_le_bytes(buf[8..16].try_into().unwrap());
            let param_size = u64::from_le_bytes(buf[16..24].try_into().unwrap());
            let mut config = [0u32; 10];
            for (j, val) in config.iter_mut().enumerate() {
                let co = 24 + j * 4;
                *val = u32::from_le_bytes(buf[co..co + 4].try_into().unwrap());
            }

            layers.push(LayerDescriptor {
                layer_type,
                param_offset,
                param_size,
                config,
            });
        }

        let weights_base = if weights_start < mmap.len() {
            mmap[weights_start..].as_ptr()
        } else {
            mmap.as_ptr() // no weights, pointer won't be dereferenced
        };
        let weights_len = mmap.len() - weights_start;

        Ok(Self {
            _mmap: mmap,
            layers,
            weights_base,
            weights_len,
        })
    }

    /// Get weight data for a layer as f32 slice (zero-copy from mmap)
    pub fn layer_weights_f32(&self, layer: &LayerDescriptor) -> &[f32] {
        let offset = layer.param_offset as usize;
        let count = layer.param_size as usize / 4;
        assert!(
            offset + count * 4 <= self.weights_len,
            "Weight access out of bounds"
        );
        // SAFETY: data is within mmap bounds, f32 alignment is handled by copy if needed
        unsafe {
            let ptr = self.weights_base.add(offset) as *const f32;
            std::slice::from_raw_parts(ptr, count)
        }
    }
}

/// Build .ocnn binary data in memory (for tests and conversion tools).
pub fn build_ocnn(layers: &[(LayerDescriptor, &[u8])]) -> Vec<u8> {
    let desc_size = 64;
    let header_size = 12 + layers.len() * desc_size;

    // Calculate total weight size
    let total_weights: usize = layers.iter().map(|(_, w)| w.len()).sum();
    let mut buf = vec![0u8; header_size + total_weights];

    // Header
    buf[0..4].copy_from_slice(MAGIC);
    buf[4..8].copy_from_slice(&VERSION.to_le_bytes());
    buf[8..12].copy_from_slice(&(layers.len() as u32).to_le_bytes());

    // Layer descriptors + weight data
    let mut weight_offset = 0u64;
    for (i, (desc, weights)) in layers.iter().enumerate() {
        let offset = 12 + i * desc_size;
        buf[offset] = desc.layer_type as u8;
        // bytes 1..8: reserved
        buf[offset + 8..offset + 16].copy_from_slice(&weight_offset.to_le_bytes());
        buf[offset + 16..offset + 24].copy_from_slice(&(weights.len() as u64).to_le_bytes());
        for (j, &val) in desc.config.iter().enumerate() {
            let co = offset + 24 + j * 4;
            buf[co..co + 4].copy_from_slice(&val.to_le_bytes());
        }

        // Copy weight data
        let w_start = header_size + weight_offset as usize;
        buf[w_start..w_start + weights.len()].copy_from_slice(weights);
        weight_offset += weights.len() as u64;
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use memmap2::MmapMut;

    fn make_mmap(data: &[u8]) -> Mmap {
        let mut mm = MmapMut::map_anon(data.len()).unwrap();
        mm.copy_from_slice(data);
        mm.make_read_only().unwrap()
    }

    #[test]
    fn test_parse_empty_model() {
        let relu_desc = LayerDescriptor {
            layer_type: LayerType::ReLU,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
        };
        let data = build_ocnn(&[(relu_desc, &[])]);
        let model = OcnnModel::from_mmap(make_mmap(&data)).unwrap();
        assert_eq!(model.layers.len(), 1);
        assert_eq!(model.layers[0].layer_type, LayerType::ReLU);
    }

    #[test]
    fn test_parse_with_weights() {
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let weight_bytes: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();

        let desc = LayerDescriptor {
            layer_type: LayerType::Linear,
            param_offset: 0,
            param_size: weight_bytes.len() as u64,
            config: [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        };
        let data = build_ocnn(&[(desc, &weight_bytes)]);
        let model = OcnnModel::from_mmap(make_mmap(&data)).unwrap();
        assert_eq!(model.layers.len(), 1);
        let w = model.layer_weights_f32(&model.layers[0]);
        assert_eq!(w, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_invalid_magic() {
        let data = b"BADM\x01\x00\x00\x00\x00\x00\x00\x00";
        let result = OcnnModel::from_mmap(make_mmap(data));
        assert!(result.is_err());
    }

    #[test]
    fn test_too_small() {
        let data = b"OCNN";
        let result = OcnnModel::from_mmap(make_mmap(data));
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_layer() {
        let relu = LayerDescriptor {
            layer_type: LayerType::ReLU,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
        };
        let hs = LayerDescriptor {
            layer_type: LayerType::HardSwish,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
        };
        let data = build_ocnn(&[(relu, &[]), (hs, &[])]);
        let model = OcnnModel::from_mmap(make_mmap(&data)).unwrap();
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.layers[0].layer_type, LayerType::ReLU);
        assert_eq!(model.layers[1].layer_type, LayerType::HardSwish);
    }
}
