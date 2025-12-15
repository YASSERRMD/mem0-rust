//! # mem0-rust
//!
//! A Rust implementation of mem0 - Universal memory layer for AI Agents.
//!
//! This library provides a flexible memory system with support for multiple:
//! - Embedding providers (OpenAI, Ollama, HuggingFace)
//! - Vector stores (In-memory, Qdrant, PostgreSQL, Redis)
//! - LLM providers (OpenAI, Ollama, Anthropic)
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use mem0_rust::{Memory, MemoryConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = MemoryConfig::default();
//!     let memory = Memory::new(config).await?;
//!
//!     // Add a memory
//!     let result = memory.add("User prefers dark mode", Default::default()).await?;
//!
//!     // Search memories
//!     let results = memory.search("user preferences", Default::default()).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod embeddings;
pub mod errors;
pub mod history;
pub mod llms;
pub mod memory;
pub mod models;
pub mod rerankers;
pub mod utils;
pub mod vector_stores;

// Re-export main types for convenience
// Re-export main types for convenience
pub use config::{
    EmbedderConfig, HuggingFaceEmbedderConfig, LLMConfig, MemoryConfig, MockEmbedderConfig,
    RerankerConfig, CohereRerankerConfig, VectorStoreConfig,
};
pub use errors::MemoryError;
pub use memory::Memory;
pub use models::{
    AddOptions, AddResult, Filters, GetAllOptions, HistoryEntry, MemoryRecord, Message, Role, SearchOptions,
    SearchResult,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::config::*;
    pub use crate::errors::*;
    pub use crate::memory::Memory;
    pub use crate::models::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_creation() {
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await;
        assert!(memory.is_ok()); 
    }
}
