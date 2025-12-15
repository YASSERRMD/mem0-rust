//! Configuration types for mem0-rust.
//!
//! This module provides comprehensive configuration options for:
//! - Embedding providers
//! - Vector store backends
//! - LLM providers
//! - Memory behavior

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for the Memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Embedding provider configuration
    pub embedder: EmbedderConfig,

    /// Vector store backend configuration
    pub vector_store: VectorStoreConfig,

    /// LLM provider configuration (optional - for inference mode)
    pub llm: Option<LLMConfig>,

    /// Path to SQLite database for history tracking
    pub history_db_path: Option<PathBuf>,

    /// Custom prompts for fact extraction
    pub custom_prompts: Option<CustomPrompts>,

    /// Reranker configuration
    pub reranker: Option<RerankerConfig>,

    /// API version
    pub version: String,

    /// Collection/index name for vector store
    pub collection_name: String,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            embedder: EmbedderConfig::default(),
            vector_store: VectorStoreConfig::default(),
            llm: None,
            history_db_path: None,
            custom_prompts: None,
            reranker: None,
            version: "1.1".to_string(),
            collection_name: "mem0".to_string(),
        }
    }
}

/// Embedding provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum EmbedderConfig {
    /// Mock embedder for testing (hash-based)
    Mock(MockEmbedderConfig),

    /// OpenAI embeddings
    #[cfg(feature = "openai")]
    OpenAI(OpenAIEmbedderConfig),

    /// Ollama local embeddings
    #[cfg(feature = "ollama")]
    Ollama(OllamaEmbedderConfig),

    /// HuggingFace Inference API embeddings
    HuggingFace(HuggingFaceEmbedderConfig),
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        EmbedderConfig::Mock(MockEmbedderConfig::default())
    }
}

/// Mock embedder configuration (for testing)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockEmbedderConfig {
    /// Embedding dimension
    pub dimensions: usize,
}

impl Default for MockEmbedderConfig {
    fn default() -> Self {
        Self { dimensions: 128 }
    }
}

/// OpenAI embedder configuration
#[cfg(feature = "openai")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIEmbedderConfig {
    /// API key (defaults to OPENAI_API_KEY env var)
    pub api_key: Option<String>,

    /// Model name
    pub model: String,

    /// Embedding dimensions (for models that support it)
    pub dimensions: Option<usize>,

    /// Base URL for API
    pub base_url: Option<String>,
}

#[cfg(feature = "openai")]
impl Default for OpenAIEmbedderConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            model: "text-embedding-3-small".to_string(),
            dimensions: Some(1536),
            base_url: None,
        }
    }
}

/// Ollama embedder configuration
#[cfg(feature = "ollama")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaEmbedderConfig {
    /// Model name
    pub model: String,

    /// Ollama server URL
    pub base_url: String,

    /// Embedding dimensions
    pub dimensions: usize,
}

#[cfg(feature = "ollama")]
impl Default for OllamaEmbedderConfig {
    fn default() -> Self {
        Self {
            model: "nomic-embed-text".to_string(),
            base_url: "http://localhost:11434".to_string(),
            dimensions: 768,
        }
    }
}

/// HuggingFace embedder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceEmbedderConfig {
    /// API key (defaults to HF_TOKEN env var)
    pub api_key: Option<String>,

    /// Model name
    pub model: String,

    /// Embedding dimensions
    pub dimensions: usize,

    /// API endpoint (optional)
    pub api_url: Option<String>,
}

impl Default for HuggingFaceEmbedderConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            dimensions: 384,
            api_url: None,
        }
    }
}

/// Vector store backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum VectorStoreConfig {
    /// In-memory vector store (default)
    Memory(MemoryStoreConfig),

    /// Qdrant vector database
    #[cfg(feature = "qdrant")]
    Qdrant(QdrantConfig),

    /// PostgreSQL with pgvector
    #[cfg(feature = "postgres")]
    Postgres(PostgresConfig),

    /// Redis with vector search
    #[cfg(feature = "redis")]
    Redis(RedisConfig),
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        VectorStoreConfig::Memory(MemoryStoreConfig::default())
    }
}

/// In-memory store configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryStoreConfig {
    /// Maximum number of entries to store
    pub max_entries: Option<usize>,
}

/// Qdrant configuration
#[cfg(feature = "qdrant")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConfig {
    /// Qdrant server URL
    pub url: String,

    /// API key (optional)
    pub api_key: Option<String>,

    /// Collection name
    pub collection_name: String,

    /// Vector dimensions
    pub dimensions: usize,

    /// Distance metric
    pub distance: DistanceMetric,
}

#[cfg(feature = "qdrant")]
impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_string(),
            api_key: None,
            collection_name: "mem0".to_string(),
            dimensions: 1536,
            distance: DistanceMetric::Cosine,
        }
    }
}

/// PostgreSQL with pgvector configuration
#[cfg(feature = "postgres")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgresConfig {
    /// Connection URL
    pub connection_url: String,

    /// Table name
    pub table_name: String,

    /// Vector dimensions
    pub dimensions: usize,
}

#[cfg(feature = "postgres")]
impl Default for PostgresConfig {
    fn default() -> Self {
        Self {
            connection_url: "postgres://localhost/mem0".to_string(),
            table_name: "memories".to_string(),
            dimensions: 1536,
        }
    }
}

/// Redis configuration
#[cfg(feature = "redis")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis connection URL
    pub url: String,

    /// Index name
    pub index_name: String,

    /// Vector dimensions
    pub dimensions: usize,
}

#[cfg(feature = "redis")]
impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://localhost:6379".to_string(),
            index_name: "mem0_idx".to_string(),
            dimensions: 1536,
        }
    }
}

/// Distance metric for vector similarity
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    #[default]
    Cosine,
    Euclidean,
    DotProduct,
}

/// LLM provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum LLMConfig {
    /// OpenAI GPT models
    #[cfg(feature = "openai")]
    OpenAI(OpenAILLMConfig),

    /// Ollama local models
    #[cfg(feature = "ollama")]
    Ollama(OllamaLLMConfig),

    /// Anthropic Claude
    #[cfg(feature = "anthropic")]
    Anthropic(AnthropicConfig),
}

/// OpenAI LLM configuration
#[cfg(feature = "openai")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAILLMConfig {
    /// API key (defaults to OPENAI_API_KEY env var)
    pub api_key: Option<String>,

    /// Model name
    pub model: String,

    /// Temperature
    pub temperature: f32,

    /// Max tokens
    pub max_tokens: Option<u32>,

    /// Base URL
    pub base_url: Option<String>,
}

#[cfg(feature = "openai")]
impl Default for OpenAILLMConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            model: "gpt-4o-mini".to_string(),
            temperature: 0.0,
            max_tokens: Some(1500),
            base_url: None,
        }
    }
}

/// Ollama LLM configuration
#[cfg(feature = "ollama")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaLLMConfig {
    /// Model name
    pub model: String,

    /// Ollama server URL
    pub base_url: String,

    /// Temperature
    pub temperature: f32,
}

#[cfg(feature = "ollama")]
impl Default for OllamaLLMConfig {
    fn default() -> Self {
        Self {
            model: "llama3.2".to_string(),
            base_url: "http://localhost:11434".to_string(),
            temperature: 0.0,
        }
    }
}

/// Anthropic configuration
#[cfg(feature = "anthropic")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    /// API key (defaults to ANTHROPIC_API_KEY env var)
    pub api_key: Option<String>,

    /// Model name
    pub model: String,

    /// Temperature
    pub temperature: f32,

    /// Max tokens
    pub max_tokens: u32,
}

#[cfg(feature = "anthropic")]
impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            model: "claude-3-haiku-20240307".to_string(),
            temperature: 0.0,
            max_tokens: 1500,
        }
    }
}

/// Custom prompts configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CustomPrompts {
    /// Custom fact extraction prompt
    pub fact_extraction: Option<String>,

    /// Custom memory update prompt
    pub memory_update: Option<String>,
}

/// Reranker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum RerankerConfig {
    /// Cohere reranker
    Cohere(CohereRerankerConfig),
}

/// Cohere reranker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereRerankerConfig {
    /// API key (defaults to COHERE_API_KEY env var)
    pub api_key: Option<String>,
    /// Model name
    pub model: String,
}

impl Default for CohereRerankerConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            model: "rerank-english-v3.0".to_string(),
        }
    }
}

