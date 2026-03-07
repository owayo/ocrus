pub mod ccl;
pub mod projection;
pub mod quality_gate;
pub mod vertical;

pub use ccl::detect_lines_ccl;
pub use projection::detect_lines_projection;
pub use quality_gate::{ImageQuality, assess_quality, should_use_fast_path};
pub use vertical::{TextOrientation, detect_columns_vertical, detect_orientation};
