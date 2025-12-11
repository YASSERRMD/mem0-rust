#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub embedding_dim: usize,
    pub max_results: usize,
    pub similarity_threshold: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            max_results: 10,
            similarity_threshold: 0.2,
        }
    }
}
