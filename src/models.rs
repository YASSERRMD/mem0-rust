use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: Uuid,
    pub content: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

impl MemoryRecord {
    pub fn new(content: impl Into<String>, metadata: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            metadata,
            created_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScoredMemory {
    pub record: MemoryRecord,
    pub score: f32,
}
