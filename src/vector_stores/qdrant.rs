//! Qdrant vector store backend.

use async_trait::async_trait;
use qdrant_client::qdrant::{
    points_selector::PointsSelectorOneOf, Condition, CreateCollectionBuilder, Distance, Filter,
    PointId, PointStruct, PointsIdsList, PointsSelector, ScrollPointsBuilder, SearchPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder, WithPayloadSelector,
};
use qdrant_client::Qdrant;
use std::collections::HashMap;

use super::traits::{VectorSearchResult, VectorStore};
use crate::config::QdrantConfig;
use crate::errors::VectorStoreError;
use crate::models::{Filters, Payload};

/// Qdrant vector store
pub struct QdrantStore {
    client: Qdrant,
    collection_name: String,
    dimensions: usize,
}

impl QdrantStore {
    /// Create a new Qdrant store
    pub async fn new(
        config: QdrantConfig,
        collection_name: &str,
        dimensions: usize,
    ) -> Result<Self, VectorStoreError> {
        let client = Qdrant::from_url(&config.url)
            .api_key(config.api_key)
            .build()
            .map_err(|e| VectorStoreError::Connection(e.to_string()))?;

        let store = Self {
            client,
            collection_name: collection_name.to_string(),
            dimensions,
        };

        // Ensure collection exists
        if !store.collection_exists().await? {
            store.create_collection().await?;
        }

        Ok(store)
    }

    /// Convert payload to Qdrant JSON
    fn payload_to_qdrant(payload: &Payload) -> HashMap<String, qdrant_client::qdrant::Value> {
        let json = serde_json::to_value(payload).unwrap_or(serde_json::Value::Null);
        let mut result = HashMap::new();

        if let serde_json::Value::Object(map) = json {
            for (k, v) in map {
                result.insert(k, Self::json_to_qdrant_value(v));
            }
        }

        result
    }

    /// Convert JSON value to Qdrant value
    fn json_to_qdrant_value(value: serde_json::Value) -> qdrant_client::qdrant::Value {
        use qdrant_client::qdrant::value::Kind;
        use qdrant_client::qdrant::Value;

        let kind = match value {
            serde_json::Value::Null => Kind::NullValue(0),
            serde_json::Value::Bool(b) => Kind::BoolValue(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Kind::IntegerValue(i)
                } else if let Some(f) = n.as_f64() {
                    Kind::DoubleValue(f)
                } else {
                    Kind::NullValue(0)
                }
            }
            serde_json::Value::String(s) => Kind::StringValue(s),
            serde_json::Value::Array(arr) => {
                let values: Vec<Value> = arr.into_iter().map(Self::json_to_qdrant_value).collect();
                Kind::ListValue(qdrant_client::qdrant::ListValue { values })
            }
            serde_json::Value::Object(map) => {
                let fields: HashMap<String, Value> = map
                    .into_iter()
                    .map(|(k, v)| (k, Self::json_to_qdrant_value(v)))
                    .collect();
                Kind::StructValue(qdrant_client::qdrant::Struct { fields })
            }
        };

        Value { kind: Some(kind) }
    }

    /// Convert Qdrant payload to Payload
    fn qdrant_to_payload(
        payload: HashMap<String, qdrant_client::qdrant::Value>,
    ) -> Result<Payload, VectorStoreError> {
        let mut json_map = serde_json::Map::new();
        for (k, v) in payload {
            json_map.insert(k, Self::qdrant_value_to_json(v));
        }
        let json = serde_json::Value::Object(json_map);
        serde_json::from_value(json)
            .map_err(|e| VectorStoreError::Search(format!("Failed to parse payload: {}", e)))
    }

    /// Convert Qdrant value to JSON
    fn qdrant_value_to_json(value: qdrant_client::qdrant::Value) -> serde_json::Value {
        use qdrant_client::qdrant::value::Kind;

        match value.kind {
            Some(Kind::NullValue(_)) => serde_json::Value::Null,
            Some(Kind::BoolValue(b)) => serde_json::Value::Bool(b),
            Some(Kind::IntegerValue(i)) => serde_json::json!(i),
            Some(Kind::DoubleValue(d)) => serde_json::json!(d),
            Some(Kind::StringValue(s)) => serde_json::Value::String(s),
            Some(Kind::ListValue(list)) => {
                let arr: Vec<serde_json::Value> = list
                    .values
                    .into_iter()
                    .map(Self::qdrant_value_to_json)
                    .collect();
                serde_json::Value::Array(arr)
            }
            Some(Kind::StructValue(s)) => {
                let mut map = serde_json::Map::new();
                for (k, v) in s.fields {
                    map.insert(k, Self::qdrant_value_to_json(v));
                }
                serde_json::Value::Object(map)
            }
            None => serde_json::Value::Null,
        }
    }

    /// Build Qdrant filter from Filters
    fn build_filter(_filters: &Filters) -> Option<Filter> {
        // TODO: Implement full filter conversion
        None
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    async fn insert(
        &self,
        id: &str,
        embedding: Vec<f32>,
        payload: Payload,
    ) -> Result<(), VectorStoreError> {
        let point = PointStruct::new(
            id.to_string(),
            embedding,
            Self::payload_to_qdrant(&payload),
        );

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, vec![point]).wait(true))
            .await
            .map_err(|e| VectorStoreError::Insert(e.to_string()))?;

        Ok(())
    }

    async fn search(
        &self,
        embedding: &[f32],
        limit: usize,
        filters: Option<&Filters>,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        let mut builder = SearchPointsBuilder::new(&self.collection_name, embedding, limit as u64)
            .with_payload(true);

        if let Some(f) = filters {
            if let Some(filter) = Self::build_filter(f) {
                builder = builder.filter(filter);
            }
        }

        let results = self
            .client
            .search_points(builder)
            .await
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        results
            .result
            .into_iter()
            .map(|point| {
                let id = match point.id {
                    Some(PointId { point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u)) }) => u,
                    Some(PointId { point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) }) => n.to_string(),
                    _ => String::new(),
                };
                let payload = Self::qdrant_to_payload(point.payload)?;
                Ok(VectorSearchResult {
                    id,
                    score: point.score,
                    payload,
                })
            })
            .collect()
    }

    async fn get(&self, id: &str) -> Result<Option<VectorSearchResult>, VectorStoreError> {
        let results = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection_name)
                    .with_payload(true)
                    .filter(Filter::must([Condition::has_id([PointId::from(
                        id.to_string(),
                    )])])),
            )
            .await
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        if let Some(point) = results.result.into_iter().next() {
            let id = match point.id {
                Some(PointId { point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u)) }) => u,
                Some(PointId { point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) }) => n.to_string(),
                _ => String::new(),
            };
            let payload = Self::qdrant_to_payload(point.payload)?;
            Ok(Some(VectorSearchResult {
                id,
                score: 1.0,
                payload,
            }))
        } else {
            Ok(None)
        }
    }

    async fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        self.client
            .delete_points(
                PointsSelector {
                    points_selector_one_of: Some(PointsSelectorOneOf::Points(PointsIdsList {
                        ids: vec![PointId::from(id.to_string())],
                    })),
                },
                Some(&self.collection_name),
            )
            .await
            .map_err(|e| VectorStoreError::Delete(e.to_string()))?;

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
            // Need to fetch existing embedding
            // For now, just use zeros
            vec![0.0; self.dimensions]
        };

        self.insert(id, emb, payload).await
    }

    async fn list(
        &self,
        filters: Option<&Filters>,
        limit: usize,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        let mut builder =
            ScrollPointsBuilder::new(&self.collection_name).with_payload(true).limit(limit as u32);

        if let Some(f) = filters {
            if let Some(filter) = Self::build_filter(f) {
                builder = builder.filter(filter);
            }
        }

        let results = self
            .client
            .scroll(builder)
            .await
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        results
            .result
            .into_iter()
            .map(|point| {
                let id = match point.id {
                    Some(PointId { point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u)) }) => u,
                    Some(PointId { point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) }) => n.to_string(),
                    _ => String::new(),
                };
                let payload = Self::qdrant_to_payload(point.payload)?;
                Ok(VectorSearchResult {
                    id,
                    score: 1.0,
                    payload,
                })
            })
            .collect()
    }

    async fn delete_all(&self, _filters: Option<&Filters>) -> Result<usize, VectorStoreError> {
        // Delete all points - this recreates the collection
        self.client
            .delete_collection(&self.collection_name)
            .await
            .map_err(|e| VectorStoreError::Delete(e.to_string()))?;

        self.create_collection().await?;
        Ok(0) // Can't easily get count before deletion
    }

    async fn collection_exists(&self) -> Result<bool, VectorStoreError> {
        self.client
            .collection_exists(&self.collection_name)
            .await
            .map_err(|e| VectorStoreError::Collection(e.to_string()))
    }

    async fn create_collection(&self) -> Result<(), VectorStoreError> {
        self.client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection_name).vectors_config(
                    VectorParamsBuilder::new(self.dimensions as u64, Distance::Cosine),
                ),
            )
            .await
            .map_err(|e| VectorStoreError::Collection(e.to_string()))?;

        Ok(())
    }
}
