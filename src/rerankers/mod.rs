mod cohere;

pub use cohere::CohereReranker;

use async_trait::async_trait;
use crate::models::ScoredMemory;
use crate::errors::MemoryError;
use crate::config::RerankerConfig;
use std::sync::Arc;

#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(&self, query: &str, results: Vec<ScoredMemory>) -> Result<Vec<ScoredMemory>, MemoryError>;
    fn model_name(&self) -> &str;
}

pub fn create_reranker(config: &RerankerConfig) -> Result<Arc<dyn Reranker>, MemoryError> {
    match config {
        RerankerConfig::Cohere(cfg) => Ok(Arc::new(CohereReranker::new(cfg.clone())?)),
    }
}
