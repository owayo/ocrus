pub mod binarize;
pub mod grayscale;
pub mod jpeg;
pub mod normalize;

pub use binarize::{binarize_adaptive, binarize_otsu, binarize_sauvola};
pub use grayscale::to_grayscale;
pub use jpeg::{is_jpeg, try_decode_jpeg};
pub use normalize::{normalize_line, normalize_line_vertical};
