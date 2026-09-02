//! The `.ocnn` model format and its executor.
//!
//! v3 replaces the v1/v2 container. The differences that matter:
//!
//! - **Named, typed op parameters** instead of an anonymous `[u32; 10]`
//! - **SSA values** instead of "layer index, negative means constant"
//! - **Symbolic dynamic dimensions**, so the Shape/Gather/Slice subgraphs that only
//!   computed shapes are folded away at conversion time instead of running as f32 tensors
//! - **Recorded golden outputs**, so a wrong artifact fails loudly instead of quietly

pub mod exec;
pub mod format;
pub mod verify;

pub use format::{Header, Model, ModelMeta, NodeDesc, Op, TensorDesc};
pub use verify::{VerifyReport, verify};
