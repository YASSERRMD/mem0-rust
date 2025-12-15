//! Mock embedder for testing.
//!
//! Uses hash-based embeddings that are deterministic but not semantic.

use async_trait::async_trait;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::traits::Embedder;
use crate::errors::EmbeddingError;

/// Hash-based mock embedder for testing
pub struct MockEmbedder {
    dimensions: usize,
}

impl MockEmbedder {
    /// Create a new mock embedder with the specified dimensions
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions: dimensions.max(1),
        }
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut vector = vec![0.0f32; self.dimensions];

        for token in text.split_whitespace() {
            let mut hasher = DefaultHasher::new();
            token.to_lowercase().hash(&mut hasher);
            let hash = hasher.finish();
            let idx = (hash as usize) % self.dimensions;
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
            let magnitude = 1.0 + ((hash >> 1) as f32 / u64::MAX as f32);
            vector[idx] += sign * magnitude;
        }

        // Normalize
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }

        Ok(vector)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        "mock-hash-embedder"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_embedder() {
        let embedder = MockEmbedder::new(128);
        let embedding = embedder.embed("Hello world").await.unwrap();
        assert_eq!(embedding.len(), 128);

        // Check normalization
        let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_deterministic() {
        let embedder = MockEmbedder::new(64);
        let e1 = embedder.embed("test").await.unwrap();
        let e2 = embedder.embed("test").await.unwrap();
        assert_eq!(e1, e2);
    }
}
