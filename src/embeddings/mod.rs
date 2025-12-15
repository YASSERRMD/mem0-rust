//! Embedding providers for mem0-rust.
//!
//! This module provides various embedding backends:
//! - Mock (hash-based, for testing)
//! - OpenAI (text-embedding-3-small/large)
//! - Ollama (local models)
//! - HuggingFace (sentence-transformers and other models)

mod mock;
mod traits;
mod huggingface;

pub use mock::MockEmbedder;
pub use traits::Embedder;
pub use huggingface::HuggingFaceEmbedder;

#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openai")]
pub use openai::OpenAIEmbedder;

#[cfg(feature = "ollama")]
mod ollama;
#[cfg(feature = "ollama")]
pub use ollama::OllamaEmbedder;

use crate::config::EmbedderConfig;
use crate::errors::EmbeddingError;
use std::sync::Arc;

/// Create an embedder from configuration
pub fn create_embedder(config: &EmbedderConfig) -> Result<Arc<dyn Embedder>, EmbeddingError> {
    match config {
        EmbedderConfig::Mock(cfg) => Ok(Arc::new(MockEmbedder::new(cfg.dimensions))),

        #[cfg(feature = "openai")]
        EmbedderConfig::OpenAI(cfg) => Ok(Arc::new(OpenAIEmbedder::new(cfg.clone())?)),

        #[cfg(feature = "ollama")]
        EmbedderConfig::Ollama(cfg) => Ok(Arc::new(OllamaEmbedder::new(cfg.clone()))),

        EmbedderConfig::HuggingFace(cfg) => {
            Ok(Arc::new(HuggingFaceEmbedder::new(cfg.clone())?))
        }
    }
}
