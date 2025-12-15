# mem0-rust

[![Build Status](https://github.com/YASSERRMD/mem0-rust/actions/workflows/rust.yml/badge.svg)](https://github.com/YASSERRMD/mem0-rust/actions)
[![Crates.io](https://img.shields.io/crates/v/mem0-rust.svg)](https://crates.io/crates/mem0-rust)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A high-performance Rust implementation of [mem0](https://github.com/mem0ai/mem0), the universal memory layer for AI Agents.

maintained by [YASSERRMD](https://github.com/YASSERRMD).

## Features

- 🦀 **Pure Rust** - Fast, safe, and efficient
- 🔌 **Multiple Backends** - Support for Memory, Qdrant, PostgreSQL (pgvector), and Redis vector stores
- 🧠 **Embedding Support** - OpenAI, Ollama, HuggingFace Inference API, and Mock providers
- 🤖 **LLM Integration** - Automatic fact extraction with OpenAI, Ollama, or Anthropic
- 🔍 **Semantic Search** - Find relevant memories using vector similarity
- 🔄 **Reranking** - Improve search relevance with rerankers (e.g. Cohere)
- 📜 **History Tracking** - Track memory changes (ADD/UPDATE/DELETE) with local SQLite history
- 👥 **Multi-User** - Isolated memory spaces per user/agent/run
- 🏷️ **Metadata Filtering** - Rich filtering with operators (eq, gt, in, contains, etc.)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mem0-rust = "0.2"
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `memory-store` (default) | In-memory vector store |
| `openai` | OpenAI embeddings and LLM |
| `ollama` | Ollama local embeddings and LLM |
| `anthropic` | Anthropic Claude LLM |
| `qdrant` | Qdrant vector database |
| `postgres` | PostgreSQL with pgvector |
| `redis` | Redis with vector search |
| `full` | All features |

## Quick Start

```rust
use mem0_rust::{Memory, MemoryConfig, AddOptions, SearchOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a memory instance with default config
    let config = MemoryConfig::default();
    let memory = Memory::new(config).await?;

    // Add a memory for a user
    memory.add(
        "I love programming in Rust",
        AddOptions::for_user("alice").raw(),
    ).await?;

    // Search for relevant memories
    let results = memory.search(
        "programming languages",
        SearchOptions::for_user("alice").with_limit(5),
    ).await?;

    for r in &results.results {
        println!("{} (score: {:.3})", r.record.content, r.score);
    }

    Ok(())
}
```

## With OpenAI (Real Embeddings + LLM Inference)

```rust
use mem0_rust::{
    Memory, MemoryConfig, AddOptions, SearchOptions,
    EmbedderConfig, LLMConfig,
    config::{OpenAIEmbedderConfig, OpenAILLMConfig},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MemoryConfig {
        embedder: EmbedderConfig::OpenAI(OpenAIEmbedderConfig::default()),
        llm: Some(LLMConfig::OpenAI(OpenAILLMConfig::default())),
        ..Default::default()
    };

    let memory = Memory::new(config).await?;

    // Add with LLM inference - facts will be automatically extracted
    memory.add(
        vec![
            mem0_rust::Message::user("Hi, I'm John. I work at Google as a data scientist."),
            mem0_rust::Message::assistant("Nice to meet you, John!"),
        ],
        AddOptions {
            user_id: Some("john".to_string()),
            infer: true, // Enable LLM inference
            ..Default::default()
        },
    ).await?;

    Ok(())
}
```

## Advanced Capabilities

### History Tracking

Track changes to your memories locally using SQLite:

```rust
use mem0_rust::MemoryConfig;
use std::path::PathBuf;

let config = MemoryConfig {
    history_db_path: Some(PathBuf::from("history.db")),
    ..Default::default()
};
// Memories added/updated/deleted will now be logged
```

### Reranking

Improve search results using a reranker (like Cohere):

```rust
use mem0_rust::{MemoryConfig, RerankerConfig, CohereRerankerConfig};

let config = MemoryConfig {
    reranker: Some(RerankerConfig::Cohere(CohereRerankerConfig {
        api_key: Some("your-cohere-key".to_string()),
        ..Default::default()
    })),
    ..Default::default()
};

// Use rerank: true in SearchOptions
```

## API Reference

### Memory Methods

| Method | Description |
|--------|-------------|
| `add(messages, options)` | Add memories from text or messages |
| `search(query, options)` | Search for relevant memories (w/ optional reranking) |
| `get(id)` | Get a memory by ID |
| `get_all(options)` | List all memories with filters |
| `update(id, content)` | Update a memory's content |
| `delete(id)` | Delete a memory |
| `history(id)` | Get version history of a memory |
| `reset(options)` | Delete all memories |

### Scoping

Memories are scoped by `user_id`, `agent_id`, and/or `run_id`.

## Examples

Run the examples:

```bash
# Basic usage
cargo run --example basic_usage

# History Tracking
cargo run --example history_tracking

# Reranking (Cohere)
cargo run --example reranking

# HuggingFace Embeddings
cargo run --example huggingface_embeddings

# Vector Stores
cargo run --example qdrant_store
cargo run --example postgres_pgvector
cargo run --example redis_vector
```

## Architecture

```
mem0-rust/
├── src/
│   ├── embeddings/      # Embedders (Mock, OpenAI, Ollama, HuggingFace)
│   ├── vector_stores/   # Stores (Memory, Qdrant, Postgres, Redis)
│   ├── llms/            # LLMs (OpenAI, Ollama, Anthropic)
│   ├── history/         # History tracking (SQLite)
│   ├── rerankers/       # Rerankers (Cohere)
│   ├── memory/          # Core memory management
│   └── utils/           # Utilities
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
