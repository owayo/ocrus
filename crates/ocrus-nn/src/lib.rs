//! Pure Rust inference for the `.ocnn` model format.
//!
//! There is exactly one format and one executor. Older `.ocnn` files (the anonymous
//! `[u32; 10]` layout) are not readable: the version in the header says so, and the fix is
//! to re-convert from the source ONNX. Carrying two formats would mean carrying two sets of
//! bugs, and the conversion is cheap.

pub mod ocnn;
pub mod ops;
pub mod tensor;

pub use ocnn::exec::Executor;
pub use ocnn::format::Model;
pub use ocnn::{VerifyReport, verify};
pub use tensor::NdTensor;
