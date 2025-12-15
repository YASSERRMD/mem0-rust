//! In-memory vector store for testing and development.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::RwLock;

use super::traits::{VectorSearchResult, VectorStore};
use crate::errors::VectorStoreError;
use crate::models::{FilterLogic, FilterOperator, Filters, Payload};

/// In-memory vector store entry
struct Entry {
    embedding: Vec<f32>,
    payload: Payload,
}

/// In-memory vector store
pub struct InMemoryStore {
    entries: RwLock<HashMap<String, Entry>>,
}

impl InMemoryStore {
    /// Create a new in-memory store
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for (va, vb) in a.iter().zip(b.iter()) {
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Check if a payload matches the given filters
    fn matches_filters(payload: &Payload, filters: Option<&Filters>) -> bool {
        let Some(filters) = filters else {
            return true;
        };

        if filters.conditions.is_empty() {
            return true;
        }

        let results: Vec<bool> = filters
            .conditions
            .iter()
            .map(|cond| {
                let value = payload.metadata.get(&cond.field);
                Self::evaluate_condition(value, &cond.operator, &cond.value)
            })
            .collect();

        match filters.logic {
            FilterLogic::And => results.iter().all(|&r| r),
            FilterLogic::Or => results.iter().any(|&r| r),
        }
    }

    /// Evaluate a single filter condition
    fn evaluate_condition(
        field_value: Option<&serde_json::Value>,
        operator: &FilterOperator,
        filter_value: &serde_json::Value,
    ) -> bool {
        match operator {
            FilterOperator::Eq => field_value == Some(filter_value),
            FilterOperator::Ne => field_value != Some(filter_value),
            FilterOperator::Gt => Self::compare_values(field_value, filter_value, |a, b| a > b),
            FilterOperator::Gte => Self::compare_values(field_value, filter_value, |a, b| a >= b),
            FilterOperator::Lt => Self::compare_values(field_value, filter_value, |a, b| a < b),
            FilterOperator::Lte => Self::compare_values(field_value, filter_value, |a, b| a <= b),
            FilterOperator::In => {
                if let Some(arr) = filter_value.as_array() {
                    field_value.map(|v| arr.contains(v)).unwrap_or(false)
                } else {
                    false
                }
            }
            FilterOperator::Nin => {
                if let Some(arr) = filter_value.as_array() {
                    field_value.map(|v| !arr.contains(v)).unwrap_or(true)
                } else {
                    true
                }
            }
            FilterOperator::Contains => {
                if let (Some(field_str), Some(filter_str)) =
                    (field_value.and_then(|v| v.as_str()), filter_value.as_str())
                {
                    field_str.contains(filter_str)
                } else {
                    false
                }
            }
            FilterOperator::IContains => {
                if let (Some(field_str), Some(filter_str)) =
                    (field_value.and_then(|v| v.as_str()), filter_value.as_str())
                {
                    field_str.to_lowercase().contains(&filter_str.to_lowercase())
                } else {
                    false
                }
            }
        }
    }

    /// Compare numeric values
    fn compare_values<F>(
        field_value: Option<&serde_json::Value>,
        filter_value: &serde_json::Value,
        cmp: F,
    ) -> bool
    where
        F: Fn(f64, f64) -> bool,
    {
        let field_num = field_value.and_then(|v| v.as_f64());
        let filter_num = filter_value.as_f64();

        match (field_num, filter_num) {
            (Some(a), Some(b)) => cmp(a, b),
            _ => false,
        }
    }


}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for InMemoryStore {
    async fn insert(
        &self,
        id: &str,
        embedding: Vec<f32>,
        payload: Payload,
    ) -> Result<(), VectorStoreError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| VectorStoreError::Insert(e.to_string()))?;

        entries.insert(id.to_string(), Entry { embedding, payload });
        Ok(())
    }

    async fn search(
        &self,
        embedding: &[f32],
        limit: usize,
        filters: Option<&Filters>,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        let entries = self
            .entries
            .read()
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        let mut results: Vec<VectorSearchResult> = entries
            .iter()
            .filter(|(_, entry)| Self::matches_filters(&entry.payload, filters))
            .map(|(id, entry)| VectorSearchResult {
                id: id.clone(),
                score: Self::cosine_similarity(embedding, &entry.embedding),
                payload: entry.payload.clone(),
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<VectorSearchResult>, VectorStoreError> {
        let entries = self
            .entries
            .read()
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        Ok(entries.get(id).map(|entry| VectorSearchResult {
            id: id.to_string(),
            score: 1.0,
            payload: entry.payload.clone(),
        }))
    }

    async fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| VectorStoreError::Delete(e.to_string()))?;

        entries
            .remove(id)
            .ok_or_else(|| VectorStoreError::NotFound(id.to_string()))?;

        Ok(())
    }

    async fn update(
        &self,
        id: &str,
        embedding: Option<Vec<f32>>,
        payload: Payload,
    ) -> Result<(), VectorStoreError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| VectorStoreError::Update(e.to_string()))?;

        let entry = entries
            .get_mut(id)
            .ok_or_else(|| VectorStoreError::NotFound(id.to_string()))?;

        if let Some(emb) = embedding {
            entry.embedding = emb;
        }
        entry.payload = payload;

        Ok(())
    }

    async fn list(
        &self,
        filters: Option<&Filters>,
        limit: usize,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        let entries = self
            .entries
            .read()
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        let mut results: Vec<VectorSearchResult> = entries
            .iter()
            .filter(|(_, entry)| Self::matches_filters(&entry.payload, filters))
            .map(|(id, entry)| VectorSearchResult {
                id: id.clone(),
                score: 1.0,
                payload: entry.payload.clone(),
            })
            .collect();

        results.truncate(limit);
        Ok(results)
    }

    async fn delete_all(&self, filters: Option<&Filters>) -> Result<usize, VectorStoreError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| VectorStoreError::Delete(e.to_string()))?;

        let to_delete: Vec<String> = entries
            .iter()
            .filter(|(_, entry)| Self::matches_filters(&entry.payload, filters))
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_delete.len();
        for id in to_delete {
            entries.remove(&id);
        }

        Ok(count)
    }

    async fn collection_exists(&self) -> Result<bool, VectorStoreError> {
        Ok(true) // In-memory store always "exists"
    }

    async fn create_collection(&self) -> Result<(), VectorStoreError> {
        Ok(()) // No-op for in-memory store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

    fn create_test_payload(data: &str) -> Payload {
        Payload {
            data: data.to_string(),
            hash: "test_hash".to_string(),
            created_at: Utc::now(),
            user_id: None,
            agent_id: None,
            run_id: None,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_insert_and_get() {
        let store = InMemoryStore::new();
        let payload = create_test_payload("test content");
        let embedding = vec![0.1, 0.2, 0.3];

        store.insert("test-id", embedding, payload).await.unwrap();

        let result = store.get("test-id").await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().payload.data, "test content");
    }

    #[tokio::test]
    async fn test_search() {
        let store = InMemoryStore::new();

        store
            .insert("id1", vec![1.0, 0.0, 0.0], create_test_payload("doc1"))
            .await
            .unwrap();
        store
            .insert("id2", vec![0.0, 1.0, 0.0], create_test_payload("doc2"))
            .await
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 10, None).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "id1"); // Most similar
    }

    #[tokio::test]
    async fn test_delete() {
        let store = InMemoryStore::new();
        store
            .insert("id1", vec![1.0], create_test_payload("doc1"))
            .await
            .unwrap();

        store.delete("id1").await.unwrap();
        assert!(store.get("id1").await.unwrap().is_none());
    }
}
