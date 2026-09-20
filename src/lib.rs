pub mod app;
pub mod architecture;
pub mod cli;
pub mod dataset;
pub mod error;
pub mod inference;
pub mod model;
pub mod neural;
pub mod optimizer;
pub mod tokenizer;
pub mod training;

pub use error::{AiError, AiResult};
