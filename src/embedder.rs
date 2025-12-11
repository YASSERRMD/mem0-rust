use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::config::MemoryConfig;

pub trait Embedder {
    fn embed(&self, text: &str) -> Vec<f32>;
    fn dimensions(&self) -> usize;
}

pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(config: &MemoryConfig) -> Self {
        Self {
            dim: config.embedding_dim,
        }
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0_f32; self.dim.max(1)];

        for token in text.split_whitespace() {
            let mut hasher = DefaultHasher::new();
            token.to_lowercase().hash(&mut hasher);
            let hash = hasher.finish();
            let idx = (hash as usize) % self.dim.max(1);
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
            let magnitude = 1.0 + ((hash >> 1) as f32 / u64::MAX as f32);
            vector[idx] += sign * magnitude;
        }

        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }

        vector
    }

    fn dimensions(&self) -> usize {
        self.dim
    }
}
