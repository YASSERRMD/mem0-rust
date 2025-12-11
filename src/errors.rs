use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("memory with id {0} not found")]
    NotFound(String),
}
