use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GraphNodeKind {
    Entity,
    Concept,
    Event,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: Uuid,
    pub kind: GraphNodeKind,
    pub value: String,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl GraphNode {
    pub fn new(kind: GraphNodeKind, value: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            kind,
            value: value.into(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: Uuid,
    pub source: Uuid,
    pub target: Uuid,
    pub relation: String,
    pub weight: f32,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl GraphEdge {
    pub fn new(source: Uuid, target: Uuid, relation: impl Into<String>, weight: f32) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            source,
            target,
            relation: relation.into(),
            weight,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}
