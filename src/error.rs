use std::{fmt, io};

#[derive(Debug)]
pub enum AiError {
    Io(io::Error),
    Model(String),
    Tokenizer(String),
    Dataset(String),
    Training(String),
    Checkpoint(String),
    Inference(String),
    Storage(String),
    Configuration(String),
    Serialization(String),
}

impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Model(e) => write!(f, "Model error: {e}"),
            Self::Tokenizer(e) => write!(f, "Tokenizer error: {e}"),
            Self::Dataset(e) => write!(f, "Dataset error: {e}"),
            Self::Training(e) => write!(f, "Training error: {e}"),
            Self::Checkpoint(e) => write!(f, "Checkpoint error: {e}"),
            Self::Inference(e) => write!(f, "Inference error: {e}"),
            Self::Storage(e) => write!(f, "Storage error: {e}"),
            Self::Configuration(e) => write!(f, "Configuration error: {e}"),
            Self::Serialization(e) => write!(f, "Serialization error: {e}"),
        }
    }
}

impl std::error::Error for AiError {}

impl From<io::Error> for AiError {
    fn from(value: io::Error) -> Self { Self::Io(value) }
}

impl From<serde_json::Error> for AiError {
    fn from(value: serde_json::Error) -> Self { Self::Serialization(value.to_string()) }
}

pub type AiResult<T> = Result<T, AiError>;
