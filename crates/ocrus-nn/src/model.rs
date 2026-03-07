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
    // v1 ops
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
    // v2 ops
    Add = 12,
    Mul = 13,
    Sub = 14,
    Div = 15,
    MatMul = 16,
    Sigmoid = 17,
    Softmax = 18,
    Concat = 19,
    Slice = 20,
    Squeeze = 21,
    Unsqueeze = 22,
    ReduceMean = 24,
    Pow = 25,
    Sqrt = 26,
    LayerNorm = 27,
    Gather = 29,
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
            12 => Some(Self::Add),
            13 => Some(Self::Mul),
            14 => Some(Self::Sub),
            15 => Some(Self::Div),
            16 => Some(Self::MatMul),
            17 => Some(Self::Sigmoid),
            18 => Some(Self::Softmax),
            19 => Some(Self::Concat),
            20 => Some(Self::Slice),
            21 => Some(Self::Squeeze),
            22 => Some(Self::Unsqueeze),
            24 => Some(Self::ReduceMean),
            25 => Some(Self::Pow),
            26 => Some(Self::Sqrt),
            27 => Some(Self::LayerNorm),
            29 => Some(Self::Gather),
            _ => None,
        }
    }
}

/// Layer descriptor
/// v1: 64 bytes (no input refs)
/// v2: 80 bytes (with input refs for DAG execution)
#[derive(Debug, Clone)]
pub struct LayerDescriptor {
    pub layer_type: LayerType,
    pub num_inputs: u8,
    pub param_offset: u64,
    pub param_size: u64,
    /// Layer-specific config (e.g., in_ch, out_ch, kh, kw, stride_h, stride_w, pad_h, pad_w, ...)
    pub config: [u32; 10],
    /// Input references (v2 only).
    /// >= 0: layer index, < 0: constant index (-(idx+1))
    pub inputs: [i32; 4],
}

/// Constant tensor entry in the .ocnn v2 format
#[derive(Debug, Clone)]
pub struct ConstantEntry {
    pub data_offset: u64,
    pub data_size: u64,
    pub shape: Vec<usize>,
}

/// Loaded .ocnn model (mmap-backed)
pub struct OcnnModel {
    _mmap: Mmap,
    pub version: u32,
    pub layers: Vec<LayerDescriptor>,
    pub constants: Vec<ConstantEntry>,
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
    /// Supports both v1 (64-byte descriptors) and v2 (80-byte descriptors + constant table).
    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        if mmap.len() < 12 {
            return Err(OcrusError::Model("Model file too small".into()));
        }

        if &mmap[0..4] != MAGIC {
            return Err(OcrusError::Model("Invalid magic bytes".into()));
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != 1 && version != 2 {
            return Err(OcrusError::Model(format!("Unsupported version: {version}")));
        }
        let num_layers = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;

        if version == 1 {
            return Self::parse_v1(mmap, num_layers);
        }

        // v2: header(16) + constant_table + layer_descriptors(80 each) + weights
        if mmap.len() < 16 {
            return Err(OcrusError::Model("v2 header too small".into()));
        }
        let num_constants = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;

        let const_table_start = 16;
        let const_entry_size = 32;
        let desc_start = const_table_start + num_constants * const_entry_size;
        let desc_size = 80;
        let weights_start = desc_start + num_layers * desc_size;

        if mmap.len() < weights_start {
            return Err(OcrusError::Model("Truncated v2 model".into()));
        }

        // Parse constant table
        let mut constants = Vec::with_capacity(num_constants);
        for i in 0..num_constants {
            let off = const_table_start + i * const_entry_size;
            let buf = &mmap[off..off + const_entry_size];
            let data_offset = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            let data_size = u64::from_le_bytes(buf[8..16].try_into().unwrap());
            let ndim = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
            let mut shape = Vec::with_capacity(ndim);
            for d in 0..ndim.min(3) {
                shape.push(
                    u32::from_le_bytes(buf[20 + d * 4..24 + d * 4].try_into().unwrap()) as usize,
                );
            }
            // 4th dim inferred if ndim > 3
            if ndim > 3 && !shape.is_empty() {
                let prod: usize = shape.iter().product();
                let num_f32 = data_size as usize / 4;
                if prod > 0 {
                    shape.push(num_f32 / prod);
                }
            }
            constants.push(ConstantEntry {
                data_offset,
                data_size,
                shape,
            });
        }

        // Parse v2 layer descriptors (80 bytes each)
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let offset = desc_start + i * desc_size;
            let buf = &mmap[offset..offset + desc_size];

            let layer_type = LayerType::from_u8(buf[0])
                .ok_or_else(|| OcrusError::Model(format!("Unknown layer type: {}", buf[0])))?;
            let num_inputs = buf[1];
            let param_offset = u64::from_le_bytes(buf[8..16].try_into().unwrap());
            let param_size = u64::from_le_bytes(buf[16..24].try_into().unwrap());
            let mut config = [0u32; 10];
            for (j, val) in config.iter_mut().enumerate() {
                let co = 24 + j * 4;
                *val = u32::from_le_bytes(buf[co..co + 4].try_into().unwrap());
            }
            let mut inputs = [0i32; 4];
            for (j, val) in inputs.iter_mut().enumerate() {
                let co = 64 + j * 4;
                *val = i32::from_le_bytes(buf[co..co + 4].try_into().unwrap());
            }

            layers.push(LayerDescriptor {
                layer_type,
                num_inputs,
                param_offset,
                param_size,
                config,
                inputs,
            });
        }

        let weights_base = if weights_start < mmap.len() {
            mmap[weights_start..].as_ptr()
        } else {
            mmap.as_ptr()
        };
        let weights_len = mmap.len() - weights_start;

        Ok(Self {
            _mmap: mmap,
            version: 2,
            layers,
            constants,
            weights_base,
            weights_len,
        })
    }

    fn parse_v1(mmap: Mmap, num_layers: usize) -> Result<Self> {
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
            let param_offset = u64::from_le_bytes(buf[8..16].try_into().unwrap());
            let param_size = u64::from_le_bytes(buf[16..24].try_into().unwrap());
            let mut config = [0u32; 10];
            for (j, val) in config.iter_mut().enumerate() {
                let co = 24 + j * 4;
                *val = u32::from_le_bytes(buf[co..co + 4].try_into().unwrap());
            }

            layers.push(LayerDescriptor {
                layer_type,
                num_inputs: 0,
                param_offset,
                param_size,
                config,
                inputs: [0; 4],
            });
        }

        let weights_base = if weights_start < mmap.len() {
            mmap[weights_start..].as_ptr()
        } else {
            mmap.as_ptr()
        };
        let weights_len = mmap.len() - weights_start;

        Ok(Self {
            _mmap: mmap,
            version: 1,
            layers,
            constants: Vec::new(),
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

    /// Get weight data for a layer as i32 slice (for extra input references in Concat >4)
    pub fn layer_weights_i32(&self, layer: &LayerDescriptor) -> &[i32] {
        let offset = layer.param_offset as usize;
        let count = layer.param_size as usize / 4;
        assert!(
            offset + count * 4 <= self.weights_len,
            "Weight access out of bounds"
        );
        // SAFETY: data is within mmap bounds
        unsafe {
            let ptr = self.weights_base.add(offset) as *const i32;
            std::slice::from_raw_parts(ptr, count)
        }
    }
}

/// Build .ocnn v1 binary data in memory (for tests and conversion tools).
pub fn build_ocnn(layers: &[(LayerDescriptor, &[u8])]) -> Vec<u8> {
    let desc_size = 64;
    let header_size = 12 + layers.len() * desc_size;

    let total_weights: usize = layers.iter().map(|(_, w)| w.len()).sum();
    let mut buf = vec![0u8; header_size + total_weights];

    buf[0..4].copy_from_slice(MAGIC);
    buf[4..8].copy_from_slice(&1u32.to_le_bytes());
    buf[8..12].copy_from_slice(&(layers.len() as u32).to_le_bytes());

    let mut weight_offset = 0u64;
    for (i, (desc, weights)) in layers.iter().enumerate() {
        let offset = 12 + i * desc_size;
        buf[offset] = desc.layer_type as u8;
        buf[offset + 8..offset + 16].copy_from_slice(&weight_offset.to_le_bytes());
        buf[offset + 16..offset + 24].copy_from_slice(&(weights.len() as u64).to_le_bytes());
        for (j, &val) in desc.config.iter().enumerate() {
            let co = offset + 24 + j * 4;
            buf[co..co + 4].copy_from_slice(&val.to_le_bytes());
        }

        let w_start = header_size + weight_offset as usize;
        buf[w_start..w_start + weights.len()].copy_from_slice(weights);
        weight_offset += weights.len() as u64;
    }

    buf
}

/// Constant tensor definition for v2 builder
pub struct ConstantDef<'a> {
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

/// Build .ocnn v2 binary data in memory
pub fn build_ocnn_v2(
    constants: &[ConstantDef<'_>],
    layers: &[(LayerDescriptor, &[u8])],
) -> Vec<u8> {
    let const_entry_size = 32;
    let desc_size = 80;
    let header_size = 16 + constants.len() * const_entry_size + layers.len() * desc_size;

    let total_const_data: usize = constants.iter().map(|c| c.data.len()).sum();
    let total_layer_weights: usize = layers.iter().map(|(_, w)| w.len()).sum();
    let mut buf = vec![0u8; header_size + total_const_data + total_layer_weights];

    // Header (16 bytes)
    buf[0..4].copy_from_slice(MAGIC);
    buf[4..8].copy_from_slice(&2u32.to_le_bytes());
    buf[8..12].copy_from_slice(&(layers.len() as u32).to_le_bytes());
    buf[12..16].copy_from_slice(&(constants.len() as u32).to_le_bytes());

    // Constant table
    let mut data_offset = 0u64;
    for (i, cdef) in constants.iter().enumerate() {
        let off = 16 + i * const_entry_size;
        buf[off..off + 8].copy_from_slice(&data_offset.to_le_bytes());
        buf[off + 8..off + 16].copy_from_slice(&(cdef.data.len() as u64).to_le_bytes());
        buf[off + 16..off + 20].copy_from_slice(&(cdef.shape.len() as u32).to_le_bytes());
        for (d, &dim) in cdef.shape.iter().take(3).enumerate() {
            let co = off + 20 + d * 4;
            buf[co..co + 4].copy_from_slice(&(dim as u32).to_le_bytes());
        }
        let w_start = header_size + data_offset as usize;
        buf[w_start..w_start + cdef.data.len()].copy_from_slice(cdef.data);
        data_offset += cdef.data.len() as u64;
    }

    // Layer descriptors (80 bytes each)
    let desc_base = 16 + constants.len() * const_entry_size;
    let mut weight_offset = total_const_data as u64;
    for (i, (desc, weights)) in layers.iter().enumerate() {
        let off = desc_base + i * desc_size;
        buf[off] = desc.layer_type as u8;
        buf[off + 1] = desc.num_inputs;
        buf[off + 8..off + 16].copy_from_slice(&weight_offset.to_le_bytes());
        buf[off + 16..off + 24].copy_from_slice(&(weights.len() as u64).to_le_bytes());
        for (j, &val) in desc.config.iter().enumerate() {
            let co = off + 24 + j * 4;
            buf[co..co + 4].copy_from_slice(&val.to_le_bytes());
        }
        for (j, &val) in desc.inputs.iter().enumerate() {
            let co = off + 64 + j * 4;
            buf[co..co + 4].copy_from_slice(&val.to_le_bytes());
        }
        let w_start = header_size + weight_offset as usize;
        buf[w_start..w_start + weights.len()].copy_from_slice(weights);
        weight_offset += weights.len() as u64;
    }

    buf
}

impl OcnnModel {
    /// Get constant data as f32 slice
    pub fn constant_f32(&self, entry: &ConstantEntry) -> &[f32] {
        let offset = entry.data_offset as usize;
        let count = entry.data_size as usize / 4;
        assert!(
            offset + count * 4 <= self.weights_len,
            "Constant access out of bounds"
        );
        unsafe {
            let ptr = self.weights_base.add(offset) as *const f32;
            std::slice::from_raw_parts(ptr, count)
        }
    }
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
            num_inputs: 0,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
            inputs: [0; 4],
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
            num_inputs: 0,
            param_offset: 0,
            param_size: weight_bytes.len() as u64,
            config: [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            inputs: [0; 4],
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
            num_inputs: 0,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
            inputs: [0; 4],
        };
        let hs = LayerDescriptor {
            layer_type: LayerType::HardSwish,
            num_inputs: 0,
            param_offset: 0,
            param_size: 0,
            config: [0; 10],
            inputs: [0; 4],
        };
        let data = build_ocnn(&[(relu, &[]), (hs, &[])]);
        let model = OcnnModel::from_mmap(make_mmap(&data)).unwrap();
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.layers[0].layer_type, LayerType::ReLU);
        assert_eq!(model.layers[1].layer_type, LayerType::HardSwish);
    }
}
