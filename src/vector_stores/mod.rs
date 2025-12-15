//! Vector store backends for mem0-rust.
//!
//! This module provides various vector storage backends:
//! - Memory (in-memory, for testing and development)
//! - Qdrant (production vector database)
//! - PostgreSQL with pgvector
//! - Redis with vector search

mod memory;
mod traits;

pub use memory::InMemoryStore;
pub use traits::VectorStore;

#[cfg(feature = "qdrant")]
mod qdrant;
#[cfg(feature = "qdrant")]
pub use qdrant::QdrantStore;

#[cfg(feature = "postgres")]
mod postgres;
#[cfg(feature = "postgres")]
pub use postgres::PostgresStore;

#[cfg(feature = "redis")]
mod redis;
#[cfg(feature = "redis")]
pub use self::redis::RedisStore;

use crate::config::VectorStoreConfig;
use crate::errors::VectorStoreError;
use std::sync::Arc;

/// Create a vector store from configuration
pub async fn create_vector_store(
    config: &VectorStoreConfig,
    collection_name: &str,
    dimensions: usize,
) -> Result<Arc<dyn VectorStore>, VectorStoreError> {
    match config {
        VectorStoreConfig::Memory(_) => {
            let _ = (collection_name, dimensions);
            Ok(Arc::new(InMemoryStore::new()))
        }

        #[cfg(feature = "qdrant")]
        VectorStoreConfig::Qdrant(cfg) => {
            let store = QdrantStore::new(cfg.clone(), collection_name, dimensions).await?;
            Ok(Arc::new(store))
        }

        #[cfg(feature = "postgres")]
        VectorStoreConfig::Postgres(cfg) => {
            let store = PostgresStore::new(cfg.clone(), collection_name, dimensions).await?;
            Ok(Arc::new(store))
        }

        #[cfg(feature = "redis")]
        VectorStoreConfig::Redis(cfg) => {
            let store = RedisStore::new(cfg.clone(), collection_name, dimensions).await?;
            Ok(Arc::new(store))
        }
    }
}
