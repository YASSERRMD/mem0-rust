//! Data models for mem0-rust.
//!
//! This module provides all the core data structures used throughout the library.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use uuid::Uuid;

/// A stored memory record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Unique identifier
    pub id: Uuid,

    /// Memory content
    pub content: String,

    /// Associated metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// User ID scope
    pub user_id: Option<String>,

    /// Agent ID scope
    pub agent_id: Option<String>,

    /// Run ID scope
    pub run_id: Option<String>,

    /// Content hash for deduplication
    pub hash: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl MemoryRecord {
    /// Create a new memory record
    pub fn new(content: impl Into<String>, metadata: serde_json::Value) -> Self {
        let content = content.into();
        let hash = Self::compute_hash(&content);
        let now = Utc::now();

        let metadata_map = match metadata {
            serde_json::Value::Object(map) => map.into_iter().collect(),
            _ => HashMap::new(),
        };

        Self {
            id: Uuid::new_v4(),
            content,
            metadata: metadata_map,
            user_id: None,
            agent_id: None,
            run_id: None,
            hash,
            created_at: now,
            updated_at: now,
        }
    }

    /// Create with scoping
    pub fn with_scoping(
        content: impl Into<String>,
        metadata: serde_json::Value,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
    ) -> Self {
        let mut record = Self::new(content, metadata);
        record.user_id = user_id;
        record.agent_id = agent_id;
        record.run_id = run_id;
        record
    }

    /// Compute content hash
    fn compute_hash(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Update the content and hash
    pub fn update_content(&mut self, content: impl Into<String>) {
        self.content = content.into();
        self.hash = Self::compute_hash(&self.content);
        self.updated_at = Utc::now();
    }
}

/// A memory with its similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredMemory {
    /// The memory record
    pub record: MemoryRecord,

    /// Similarity score
    pub score: f32,
}

/// Message role for chat-style input
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message
    System,
    /// User message
    User,
    /// Assistant message
    Assistant,
}

/// A chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message role
    pub role: Role,

    /// Message content
    pub content: String,

    /// Optional actor name
    pub name: Option<String>,
}

impl Message {
    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            name: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            name: None,
        }
    }

    /// Set the actor name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Input that can be converted to messages
#[derive(Debug, Clone)]
pub enum Messages {
    /// Plain text input
    Text(String),
    /// Single message
    Single(Message),
    /// Multiple messages
    Multiple(Vec<Message>),
}

impl From<&str> for Messages {
    fn from(s: &str) -> Self {
        Messages::Text(s.to_string())
    }
}

impl From<String> for Messages {
    fn from(s: String) -> Self {
        Messages::Text(s)
    }
}

impl From<Message> for Messages {
    fn from(m: Message) -> Self {
        Messages::Single(m)
    }
}

impl From<Vec<Message>> for Messages {
    fn from(v: Vec<Message>) -> Self {
        Messages::Multiple(v)
    }
}

impl Messages {
    /// Convert to a list of messages
    pub fn into_messages(self) -> Vec<Message> {
        match self {
            Messages::Text(s) => vec![Message::user(s)],
            Messages::Single(m) => vec![m],
            Messages::Multiple(v) => v,
        }
    }
}

/// Options for adding memories
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AddOptions {
    /// User ID scope
    pub user_id: Option<String>,

    /// Agent ID scope
    pub agent_id: Option<String>,

    /// Run ID scope
    pub run_id: Option<String>,

    /// Additional metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    /// Whether to use LLM for inference (default: true)
    pub infer: bool,
}

impl AddOptions {
    /// Create options with user scope
    pub fn for_user(user_id: impl Into<String>) -> Self {
        Self {
            user_id: Some(user_id.into()),
            infer: true,
            ..Default::default()
        }
    }

    /// Create options with agent scope
    pub fn for_agent(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: Some(agent_id.into()),
            infer: true,
            ..Default::default()
        }
    }

    /// Disable LLM inference
    pub fn raw(mut self) -> Self {
        self.infer = false;
        self
    }
}

/// Result of adding memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddResult {
    /// List of memory operations performed
    pub results: Vec<MemoryEvent>,
}

/// A memory operation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    /// Memory ID
    pub id: Uuid,

    /// Memory content
    pub memory: String,

    /// Event type
    pub event: EventType,
}

/// Type of memory event
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum EventType {
    /// Memory was added
    Add,
    /// Memory was updated
    Update,
    /// Memory was deleted
    Delete,
    /// No change
    Noop,
}

/// Options for searching memories
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchOptions {
    /// User ID filter
    pub user_id: Option<String>,

    /// Agent ID filter
    pub agent_id: Option<String>,

    /// Run ID filter
    pub run_id: Option<String>,

    /// Maximum number of results
    pub limit: Option<usize>,

    /// Minimum similarity threshold
    pub threshold: Option<f32>,

    /// Additional metadata filters
    pub filters: Option<Filters>,

    /// Whether to rerank results
    pub rerank: bool,
}

impl SearchOptions {
    /// Create options with user filter
    pub fn for_user(user_id: impl Into<String>) -> Self {
        Self {
            user_id: Some(user_id.into()),
            limit: Some(10),
            ..Default::default()
        }
    }

    /// Set the result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set the similarity threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Found memories with scores
    pub results: Vec<ScoredMemory>,
}

/// Metadata filters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Filters {
    /// Filter conditions
    pub conditions: Vec<FilterCondition>,

    /// Logic operator between conditions
    pub logic: FilterLogic,
}

/// A filter condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    /// Field name
    pub field: String,

    /// Operator
    pub operator: FilterOperator,

    /// Value to compare
    pub value: serde_json::Value,
}

/// Filter operators
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FilterOperator {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Greater than
    Gt,
    /// Greater than or equal
    Gte,
    /// Less than
    Lt,
    /// Less than or equal
    Lte,
    /// In list
    In,
    /// Not in list
    Nin,
    /// Contains (for strings)
    Contains,
    /// Contains (case-insensitive)
    IContains,
}

/// Logic for combining filter conditions
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum FilterLogic {
    #[default]
    And,
    Or,
}

/// Options for listing all memories
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GetAllOptions {
    /// User ID filter
    pub user_id: Option<String>,

    /// Agent ID filter
    pub agent_id: Option<String>,

    /// Run ID filter
    pub run_id: Option<String>,

    /// Maximum number of results
    pub limit: Option<usize>,
}

/// A history entry for a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// History entry ID
    pub id: Uuid,

    /// Memory ID this history belongs to
    pub memory_id: Uuid,

    /// Previous content
    pub previous_content: Option<String>,

    /// New content
    pub new_content: String,

    /// Event type
    pub event: EventType,

    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Options for resetting memories
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResetOptions {
    /// User ID scope (if set, only reset this user's memories)
    pub user_id: Option<String>,

    /// Agent ID scope
    pub agent_id: Option<String>,
}

/// Payload for vector store operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Payload {
    /// Memory content
    pub data: String,

    /// Memory hash
    pub hash: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// User ID
    pub user_id: Option<String>,

    /// Agent ID
    pub agent_id: Option<String>,

    /// Run ID
    pub run_id: Option<String>,

    /// Additional metadata
    #[serde(flatten)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl From<&MemoryRecord> for Payload {
    fn from(record: &MemoryRecord) -> Self {
        Self {
            data: record.content.clone(),
            hash: record.hash.clone(),
            created_at: record.created_at,
            user_id: record.user_id.clone(),
            agent_id: record.agent_id.clone(),
            run_id: record.run_id.clone(),
            metadata: record.metadata.clone(),
        }
    }
}
