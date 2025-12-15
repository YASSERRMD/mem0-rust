//! PostgreSQL with pgvector store backend.

use async_trait::async_trait;
use sqlx::{postgres::PgPoolOptions, PgPool, Row};

use super::traits::{VectorSearchResult, VectorStore};
use crate::config::PostgresConfig;
use crate::errors::VectorStoreError;
use crate::models::{Filters, Payload};

/// PostgreSQL with pgvector vector store
pub struct PostgresStore {
    pool: PgPool,
    table_name: String,
    dimensions: usize,
}

impl PostgresStore {
    /// Create a new PostgreSQL store
    pub async fn new(
        config: PostgresConfig,
        collection_name: &str,
        dimensions: usize,
    ) -> Result<Self, VectorStoreError> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&config.connection_url)
            .await
            .map_err(|e| VectorStoreError::Connection(e.to_string()))?;

        let table_name = format!("{}_{}", config.table_name, collection_name);

        let store = Self {
            pool,
            table_name,
            dimensions,
        };

        // Ensure pgvector extension and table exist
        store.create_collection().await?;

        Ok(store)
    }

    /// Convert Payload to JSON
    fn payload_to_json(payload: &Payload) -> Result<serde_json::Value, VectorStoreError> {
        serde_json::to_value(payload)
            .map_err(|e| VectorStoreError::Insert(format!("Failed to serialize payload: {}", e)))
    }

    /// Convert JSON to Payload
    fn json_to_payload(json: serde_json::Value) -> Result<Payload, VectorStoreError> {
        serde_json::from_value(json)
            .map_err(|e| VectorStoreError::Search(format!("Failed to deserialize payload: {}", e)))
    }

    /// Format embedding for pgvector
    fn format_embedding(embedding: &[f32]) -> String {
        format!(
            "[{}]",
            embedding
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

#[async_trait]
impl VectorStore for PostgresStore {
    async fn insert(
        &self,
        id: &str,
        embedding: Vec<f32>,
        payload: Payload,
    ) -> Result<(), VectorStoreError> {
        let embedding_str = Self::format_embedding(&embedding);
        let payload_json = Self::payload_to_json(&payload)?;

        let query = format!(
            r#"
            INSERT INTO {} (id, embedding, payload, user_id, agent_id, run_id, created_at)
            VALUES ($1, $2::vector, $3, $4, $5, $6, $7)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                payload = EXCLUDED.payload,
                user_id = EXCLUDED.user_id,
                agent_id = EXCLUDED.agent_id,
                run_id = EXCLUDED.run_id
            "#,
            self.table_name
        );

        sqlx::query(&query)
            .bind(id)
            .bind(&embedding_str)
            .bind(&payload_json)
            .bind(&payload.user_id)
            .bind(&payload.agent_id)
            .bind(&payload.run_id)
            .bind(payload.created_at)
            .execute(&self.pool)
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
        let embedding_str = Self::format_embedding(embedding);
        
        // Build WHERE clause from filters
        let mut where_clauses = Vec::new();
        
        if let Some(_f) = filters {
            // TODO: Implement full filter translation
        }
        
        let where_clause = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        let query = format!(
            r#"
            SELECT id, payload, 1 - (embedding <=> $1::vector) as score
            FROM {}
            {}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            "#,
            self.table_name, where_clause
        );

        let rows = sqlx::query(&query)
            .bind(&embedding_str)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            let id: String = row.get("id");
            let payload_json: serde_json::Value = row.get("payload");
            let score: f64 = row.get("score");

            let payload = Self::json_to_payload(payload_json)?;
            results.push(VectorSearchResult {
                id,
                score: score as f32,
                payload,
            });
        }

        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<VectorSearchResult>, VectorStoreError> {
        let query = format!(
            r#"SELECT id, payload FROM {} WHERE id = $1"#,
            self.table_name
        );

        let row = sqlx::query(&query)
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        match row {
            Some(row) => {
                let id: String = row.get("id");
                let payload_json: serde_json::Value = row.get("payload");
                let payload = Self::json_to_payload(payload_json)?;

                Ok(Some(VectorSearchResult {
                    id,
                    score: 1.0,
                    payload,
                }))
            }
            None => Ok(None),
        }
    }

    async fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        let query = format!(r#"DELETE FROM {} WHERE id = $1"#, self.table_name);

        let result = sqlx::query(&query)
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| VectorStoreError::Delete(e.to_string()))?;

        if result.rows_affected() == 0 {
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
        let payload_json = Self::payload_to_json(&payload)?;

        if let Some(emb) = embedding {
            let embedding_str = Self::format_embedding(&emb);
            let query = format!(
                r#"
                UPDATE {} SET
                    embedding = $2::vector,
                    payload = $3,
                    user_id = $4,
                    agent_id = $5,
                    run_id = $6
                WHERE id = $1
                "#,
                self.table_name
            );

            sqlx::query(&query)
                .bind(id)
                .bind(&embedding_str)
                .bind(&payload_json)
                .bind(&payload.user_id)
                .bind(&payload.agent_id)
                .bind(&payload.run_id)
                .execute(&self.pool)
                .await
                .map_err(|e| VectorStoreError::Update(e.to_string()))?;
        } else {
            let query = format!(
                r#"
                UPDATE {} SET
                    payload = $2,
                    user_id = $3,
                    agent_id = $4,
                    run_id = $5
                WHERE id = $1
                "#,
                self.table_name
            );

            sqlx::query(&query)
                .bind(id)
                .bind(&payload_json)
                .bind(&payload.user_id)
                .bind(&payload.agent_id)
                .bind(&payload.run_id)
                .execute(&self.pool)
                .await
                .map_err(|e| VectorStoreError::Update(e.to_string()))?;
        }

        Ok(())
    }

    async fn list(
        &self,
        filters: Option<&Filters>,
        limit: usize,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        let mut where_clauses = Vec::new();
        
        if let Some(_f) = filters {
            // TODO: Implement full filter translation
        }
        
        let where_clause = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        let query = format!(
            r#"SELECT id, payload FROM {} {} LIMIT $1"#,
            self.table_name, where_clause
        );

        let rows = sqlx::query(&query)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            let id: String = row.get("id");
            let payload_json: serde_json::Value = row.get("payload");
            let payload = Self::json_to_payload(payload_json)?;

            results.push(VectorSearchResult {
                id,
                score: 1.0,
                payload,
            });
        }

        Ok(results)
    }

    async fn delete_all(&self, filters: Option<&Filters>) -> Result<usize, VectorStoreError> {
        let mut where_clauses = Vec::new();
        
        if let Some(_f) = filters {
            // TODO: Implement full filter translation
        }
        
        let where_clause = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        let query = format!(r#"DELETE FROM {} {}"#, self.table_name, where_clause);

        let result = sqlx::query(&query)
            .execute(&self.pool)
            .await
            .map_err(|e| VectorStoreError::Delete(e.to_string()))?;

        Ok(result.rows_affected() as usize)
    }

    async fn collection_exists(&self) -> Result<bool, VectorStoreError> {
        let query = r#"
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = $1
            )
        "#;

        let row = sqlx::query(query)
            .bind(&self.table_name)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| VectorStoreError::Collection(e.to_string()))?;

        Ok(row.get::<bool, _>(0))
    }

    async fn create_collection(&self) -> Result<(), VectorStoreError> {
        // Enable pgvector extension
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&self.pool)
            .await
            .map_err(|e| VectorStoreError::Collection(format!("Failed to enable pgvector: {}", e)))?;

        // Create table
        let query = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                id TEXT PRIMARY KEY,
                embedding vector({}),
                payload JSONB NOT NULL,
                user_id TEXT,
                agent_id TEXT,
                run_id TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            "#,
            self.table_name, self.dimensions
        );

        sqlx::query(&query)
            .execute(&self.pool)
            .await
            .map_err(|e| VectorStoreError::Collection(e.to_string()))?;

        // Create index for vector similarity search
        let index_query = format!(
            r#"
            CREATE INDEX IF NOT EXISTS {}_embedding_idx 
            ON {} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            "#,
            self.table_name, self.table_name
        );

        // Index creation may fail if not enough rows, that's okay
        let _ = sqlx::query(&index_query).execute(&self.pool).await;

        Ok(())
    }
}
