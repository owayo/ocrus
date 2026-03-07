use thiserror::Error;

#[derive(Debug, Error)]
pub enum OcrusError {
    #[error("Image error: {0}")]
    Image(String),

    #[error("Layout error: {0}")]
    Layout(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Config error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, OcrusError>;
