//! Error types for mem0-rust.
//!
//! This module provides comprehensive error types for all operations.

use thiserror::Error;

/// Main error type for memory operations
#[derive(Error, Debug)]
pub enum MemoryError {
    /// Memory record not found
    #[error("memory with id {0} not found")]
    NotFound(String),

    /// Embedding dimension mismatch
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Embedding provider error
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    /// Vector store error
    #[error("vector store error: {0}")]
    VectorStore(#[from] VectorStoreError),

    /// LLM error
    #[error("LLM error: {0}")]
    LLM(#[from] LLMError),

    /// Configuration error
    #[error("configuration error: {0}")]
    Config(String),

    /// Serialization error
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Invalid input
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// History database error
    #[error("history database error: {0}")]
    History(String),

    /// Reranking error
    #[error("reranking error: {0}")]
    Reranker(String),
}

/// Embedding provider errors
#[derive(Error, Debug)]
pub enum EmbeddingError {
    /// API error
    #[error("API error: {0}")]
    Api(String),

    /// Network error
    #[error("network error: {0}")]
    Network(String),

    /// Rate limit exceeded
    #[error("rate limit exceeded")]
    RateLimited,

    /// Invalid response
    #[error("invalid response: {0}")]
    InvalidResponse(String),

    /// Provider not configured
    #[error("embedding provider not configured")]
    NotConfigured,
}

/// Vector store errors
#[derive(Error, Debug)]
pub enum VectorStoreError {
    /// Connection error
    #[error("connection error: {0}")]
    Connection(String),

    /// Record not found
    #[error("record not found: {0}")]
    NotFound(String),

    /// Insert error
    #[error("insert error: {0}")]
    Insert(String),

    /// Search error
    #[error("search error: {0}")]
    Search(String),

    /// Delete error
    #[error("delete error: {0}")]
    Delete(String),

    /// Update error
    #[error("update error: {0}")]
    Update(String),

    /// Collection/index error
    #[error("collection error: {0}")]
    Collection(String),

    /// Provider not configured
    #[error("vector store not configured")]
    NotConfigured,
}

/// LLM provider errors
#[derive(Error, Debug)]
pub enum LLMError {
    /// API error
    #[error("API error: {0}")]
    Api(String),

    /// Network error
    #[error("network error: {0}")]
    Network(String),

    /// Rate limit exceeded
    #[error("rate limit exceeded")]
    RateLimited,

    /// Invalid response
    #[error("invalid response: {0}")]
    InvalidResponse(String),

    /// JSON parsing error
    #[error("JSON parsing error: {0}")]
    JsonParse(String),

    /// Provider not configured
    #[error("LLM provider not configured")]
    NotConfigured,
}

/// Result type alias for memory operations
pub type Result<T> = std::result::Result<T, MemoryError>;
