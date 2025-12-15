//! Redis with vector search store backend.

use async_trait::async_trait;
use redis::{aio::ConnectionManager, AsyncCommands, Client};
use serde::{Deserialize, Serialize};

use super::traits::{VectorSearchResult, VectorStore};
use crate::config::RedisConfig;
use crate::errors::VectorStoreError;
use crate::models::{Filters, Payload};

/// Redis with vector search store
pub struct RedisStore {
    conn: ConnectionManager,
    index_name: String,
    prefix: String,
    dimensions: usize,
}

impl RedisStore {
    /// Create a new Redis store
    pub async fn new(
        config: RedisConfig,
        collection_name: &str,
        dimensions: usize,
    ) -> Result<Self, VectorStoreError> {
        let client = Client::open(config.url.as_str())
            .map_err(|e| VectorStoreError::Connection(e.to_string()))?;

        let conn = ConnectionManager::new(client)
            .await
            .map_err(|e| VectorStoreError::Connection(e.to_string()))?;

        let index_name = format!("{}_{}", config.index_name, collection_name);
        let prefix = format!("mem0:{}:", collection_name);

        let store = Self {
            conn,
            index_name,
            prefix,
            dimensions,
        };

        // Create index if it doesn't exist
        if !store.collection_exists().await? {
            store.create_collection().await?;
        }

        Ok(store)
    }

    /// Get the full key for a document
    fn doc_key(&self, id: &str) -> String {
        format!("{}{}", self.prefix, id)
    }
}

/// Stored document in Redis
#[derive(Debug, Serialize, Deserialize)]
struct RedisDocument {
    payload: Payload,
    embedding: Vec<f32>,
}

#[async_trait]
impl VectorStore for RedisStore {
    async fn insert(
        &self,
        id: &str,
        embedding: Vec<f32>,
        payload: Payload,
    ) -> Result<(), VectorStoreError> {
        let mut conn = self.conn.clone();
        let key = self.doc_key(id);

        let doc = RedisDocument {
            payload: payload.clone(),
            embedding: embedding.clone(),
        };

        let payload_json = serde_json::to_string(&doc.payload)
            .map_err(|e| VectorStoreError::Insert(e.to_string()))?;

        // Convert embedding to bytes for Redis
        let embedding_bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // Store as hash with embedding and payload
        redis::pipe()
            .hset(&key, "payload", &payload_json)
            .hset(&key, "embedding", &embedding_bytes)
            .hset(&key, "user_id", payload.user_id.as_deref().unwrap_or(""))
            .hset(&key, "agent_id", payload.agent_id.as_deref().unwrap_or(""))
            .hset(&key, "run_id", payload.run_id.as_deref().unwrap_or(""))
            .hset(&key, "data", &payload.data)
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| VectorStoreError::Insert(e.to_string()))?;

        Ok(())
    }

    async fn search(
        &self,
        embedding: &[f32],
        limit: usize,
        _filters: Option<&Filters>,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        let mut conn = self.conn.clone();

        // Convert embedding to bytes
        let embedding_bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // Build FT.SEARCH query for RediSearch
        let query = format!(
            "*=>[KNN {} @embedding $vec AS score]",
            limit
        );

        let result: redis::Value = redis::cmd("FT.SEARCH")
            .arg(&self.index_name)
            .arg(&query)
            .arg("PARAMS")
            .arg("2")
            .arg("vec")
            .arg(&embedding_bytes)
            .arg("SORTBY")
            .arg("score")
            .arg("DIALECT")
            .arg("2")
            .arg("RETURN")
            .arg("2")
            .arg("payload")
            .arg("score")
            .query_async(&mut conn)
            .await
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        // Parse Redis response
        let mut results = Vec::new();
        
        if let redis::Value::Array(arr) = result {
            let mut iter = arr.into_iter().skip(1); // Skip count
            while let (Some(redis::Value::BulkString(key_bytes)), Some(redis::Value::Array(fields))) = (iter.next(), iter.next()) {
                let key = String::from_utf8_lossy(&key_bytes);
                let id = key.strip_prefix(&self.prefix).unwrap_or(&key).to_string();
                
                let mut payload_json: Option<String> = None;
                let mut score: f32 = 0.0;
                
                let mut field_iter = fields.into_iter();
                while let (Some(redis::Value::BulkString(field_name)), Some(field_value)) = (field_iter.next(), field_iter.next()) {
                    let name = String::from_utf8_lossy(&field_name);
                    match name.as_ref() {
                        "payload" => {
                            if let redis::Value::BulkString(v) = field_value {
                                payload_json = Some(String::from_utf8_lossy(&v).to_string());
                            }
                        }
                        "score" => {
                            if let redis::Value::BulkString(v) = field_value {
                                let s = String::from_utf8_lossy(&v);
                                score = s.parse().unwrap_or(0.0);
                            }
                        }
                        _ => {}
                    }
                }
                
                if let Some(json) = payload_json {
                    if let Ok(payload) = serde_json::from_str(&json) {
                        results.push(VectorSearchResult {
                            id,
                            score: 1.0 - score, // Convert distance to similarity
                            payload,
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<VectorSearchResult>, VectorStoreError> {
        let mut conn = self.conn.clone();
        let key = self.doc_key(id);

        let payload_json: Option<String> = conn
            .hget(&key, "payload")
            .await
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        match payload_json {
            Some(json) => {
                let payload: Payload = serde_json::from_str(&json)
                    .map_err(|e| VectorStoreError::Search(e.to_string()))?;

                Ok(Some(VectorSearchResult {
                    id: id.to_string(),
                    score: 1.0,
                    payload,
                }))
            }
            None => Ok(None),
        }
    }

    async fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        let mut conn = self.conn.clone();
        let key = self.doc_key(id);

        let deleted: i32 = conn
            .del(&key)
            .await
            .map_err(|e| VectorStoreError::Delete(e.to_string()))?;

        if deleted == 0 {
            return Err(VectorStoreError::NotFound(id.to_string()));
        }

        Ok(())
    }

    async fn update(
        &self,
        id: &str,
        embedding: Option<Vec<f32>>,
        payload: Payload,
    ) -> Result<(), VectorStoreError> {
        // Get existing embedding if not provided
        let emb = if let Some(e) = embedding {
            e
        } else {
            let mut conn = self.conn.clone();
            let key = self.doc_key(id);
            
            let emb_bytes: Option<Vec<u8>> = conn
                .hget(&key, "embedding")
                .await
                .map_err(|e| VectorStoreError::Update(e.to_string()))?;
            
            match emb_bytes {
                Some(bytes) => {
                    bytes
                        .chunks(4)
                        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap_or([0; 4])))
                        .collect()
                }
                None => return Err(VectorStoreError::NotFound(id.to_string())),
            }
        };

        self.insert(id, emb, payload).await
    }

    async fn list(
        &self,
        _filters: Option<&Filters>,
        limit: usize,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        let mut conn = self.conn.clone();

        // Use SCAN to iterate through keys
        let pattern = format!("{}*", self.prefix);
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(&pattern)
            .query_async(&mut conn)
            .await
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        let mut results = Vec::new();
        for key in keys.into_iter().take(limit) {
            let id = key.strip_prefix(&self.prefix).unwrap_or(&key).to_string();
            
            let payload_json: Option<String> = conn
                .hget(&key, "payload")
                .await
                .map_err(|e| VectorStoreError::Search(e.to_string()))?;

            if let Some(json) = payload_json {
                if let Ok(payload) = serde_json::from_str(&json) {
                    results.push(VectorSearchResult {
                        id,
                        score: 1.0,
                        payload,
                    });
                }
            }
        }

        Ok(results)
    }

    async fn delete_all(&self, _filters: Option<&Filters>) -> Result<usize, VectorStoreError> {
        let mut conn = self.conn.clone();

        let pattern = format!("{}*", self.prefix);
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(&pattern)
            .query_async(&mut conn)
            .await
            .map_err(|e| VectorStoreError::Delete(e.to_string()))?;

        let count = keys.len();
        for key in keys {
            let _: () = conn
                .del(&key)
                .await
                .map_err(|e| VectorStoreError::Delete(e.to_string()))?;
        }

        Ok(count)
    }

    async fn collection_exists(&self) -> Result<bool, VectorStoreError> {
        let mut conn = self.conn.clone();

        let result: redis::Value = redis::cmd("FT._LIST")
            .query_async(&mut conn)
            .await
            .map_err(|e| VectorStoreError::Collection(e.to_string()))?;

        if let redis::Value::Array(indices) = result {
            for idx in indices {
                if let redis::Value::BulkString(name) = idx {
                    if String::from_utf8_lossy(&name) == self.index_name {
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    async fn create_collection(&self) -> Result<(), VectorStoreError> {
        let mut conn = self.conn.clone();

        // Create RediSearch index with vector field
        let result: Result<redis::Value, _> = redis::cmd("FT.CREATE")
            .arg(&self.index_name)
            .arg("ON")
            .arg("HASH")
            .arg("PREFIX")
            .arg("1")
            .arg(&self.prefix)
            .arg("SCHEMA")
            .arg("embedding")
            .arg("VECTOR")
            .arg("FLAT")
            .arg("6")
            .arg("TYPE")
            .arg("FLOAT32")
            .arg("DIM")
            .arg(self.dimensions)
            .arg("DISTANCE_METRIC")
            .arg("COSINE")
            .arg("payload")
            .arg("TEXT")
            .arg("data")
            .arg("TEXT")
            .arg("user_id")
            .arg("TAG")
            .arg("agent_id")
            .arg("TAG")
            .arg("run_id")
            .arg("TAG")
            .query_async(&mut conn)
            .await;

        match result {
            Ok(_) => Ok(()),
            Err(e) => {
                // Index might already exist
                if e.to_string().contains("Index already exists") {
                    Ok(())
                } else {
                    Err(VectorStoreError::Collection(e.to_string()))
                }
            }
        }
    }
}
