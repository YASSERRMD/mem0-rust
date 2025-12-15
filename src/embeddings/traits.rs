//! Embedder trait definition.

use async_trait::async_trait;
use crate::errors::EmbeddingError;

/// Trait for embedding text into vectors
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a single text into a vector
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Embed multiple texts into vectors
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Get the dimension of embeddings produced by this embedder
    fn dimensions(&self) -> usize;

    /// Get the model name (for logging/debugging)
    fn model_name(&self) -> &str {
        "unknown"
    }
}
