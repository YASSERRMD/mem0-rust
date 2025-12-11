use crate::config::MemoryConfig;
use crate::embedder::HashEmbedder;
use crate::errors::MemoryError;
use crate::models::{MemoryRecord, ScoredMemory};
use crate::store::{embed_and_store, search_store, InMemoryStore, VectorStore};

pub struct MemoryClient {
    config: MemoryConfig,
    embedder: HashEmbedder,
    store: InMemoryStore,
}

impl MemoryClient {
    pub fn new(config: MemoryConfig) -> Self {
        let embedder = HashEmbedder::new(&config);
        Self {
            config,
            embedder,
            store: InMemoryStore::default(),
        }
    }

    pub fn add(&mut self, content: impl Into<String>, metadata: serde_json::Value) -> MemoryRecord {
        let record = MemoryRecord::new(content, metadata);
        embed_and_store(&mut self.store, &self.embedder, record)
    }

    pub fn search(&self, query: &str) -> Result<Vec<ScoredMemory>, MemoryError> {
        let mut results =
            search_store(&self.store, &self.embedder, query, self.config.max_results)?;

        results.retain(|item| item.score >= self.config.similarity_threshold);
        Ok(results)
    }

    pub fn delete(&mut self, id: &str) -> Result<(), MemoryError> {
        self.store.delete(id)
    }
}

impl Default for MemoryClient {
    fn default() -> Self {
        Self::new(MemoryConfig::default())
    }
}
