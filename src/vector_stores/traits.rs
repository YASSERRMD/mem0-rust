//! Vector store trait definition.

use async_trait::async_trait;
use crate::errors::VectorStoreError;
use crate::models::{Filters, MemoryRecord, Payload, ScoredMemory};

/// Search result from vector store
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    /// Record ID
    pub id: String,
    /// Similarity score
    pub score: f32,
    /// Payload data
    pub payload: Payload,
}

/// Trait for vector storage backends
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert a record with its embedding
    async fn insert(
        &self,
        id: &str,
        embedding: Vec<f32>,
        payload: Payload,
    ) -> Result<(), VectorStoreError>;

    /// Search for similar vectors
    async fn search(
        &self,
        embedding: &[f32],
        limit: usize,
        filters: Option<&Filters>,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError>;

    /// Get a single record by ID
    async fn get(&self, id: &str) -> Result<Option<VectorSearchResult>, VectorStoreError>;

    /// Delete a record by ID
    async fn delete(&self, id: &str) -> Result<(), VectorStoreError>;

    /// Update a record
    async fn update(
        &self,
        id: &str,
        embedding: Option<Vec<f32>>,
        payload: Payload,
    ) -> Result<(), VectorStoreError>;

    /// List all records with optional filters
    async fn list(
        &self,
        filters: Option<&Filters>,
        limit: usize,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError>;

    /// Delete all records matching filters
    async fn delete_all(&self, filters: Option<&Filters>) -> Result<usize, VectorStoreError>;

    /// Check if collection/index exists
    async fn collection_exists(&self) -> Result<bool, VectorStoreError>;

    /// Create collection/index if it doesn't exist
    async fn create_collection(&self) -> Result<(), VectorStoreError>;
}

/// Convert vector search result to scored memory
impl VectorSearchResult {
    /// Convert to a MemoryRecord
    pub fn to_memory_record(&self) -> MemoryRecord {
        MemoryRecord {
            id: uuid::Uuid::parse_str(&self.id).unwrap_or_else(|_| uuid::Uuid::new_v4()),
            content: self.payload.data.clone(),
            metadata: self.payload.metadata.clone(),
            user_id: self.payload.user_id.clone(),
            agent_id: self.payload.agent_id.clone(),
            run_id: self.payload.run_id.clone(),
            hash: self.payload.hash.clone(),
            created_at: self.payload.created_at,
            updated_at: self.payload.created_at, // Use created_at as fallback
        }
    }

    /// Convert to ScoredMemory
    pub fn to_scored_memory(&self) -> ScoredMemory {
        ScoredMemory {
            record: self.to_memory_record(),
            score: self.score,
        }
    }
}
