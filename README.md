# mem0-rust

A Rust implementation of [mem0](https://github.com/mem0ai/mem0) - Universal memory layer for AI Agents.

## Features

- 🦀 **Pure Rust** - Fast, safe, and efficient
- 🔌 **Multiple Backends** - Support for various embedding and vector store providers
- 🤖 **LLM Integration** - Automatic fact extraction with OpenAI, Ollama, or Anthropic
- 🔍 **Semantic Search** - Find relevant memories using vector similarity
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

## API Reference

### Memory Methods

| Method | Description |
|--------|-------------|
| `add(messages, options)` | Add memories from text or messages |
| `search(query, options)` | Search for relevant memories |
| `get(id)` | Get a memory by ID |
| `get_all(options)` | List all memories with filters |
| `update(id, content)` | Update a memory's content |
| `delete(id)` | Delete a memory |
| `history(id)` | Get version history (coming soon) |
| `reset(options)` | Delete all memories |

### Scoping

Memories are scoped by `user_id`, `agent_id`, and/or `run_id`:

```rust
// User-scoped memory
AddOptions::for_user("alice")

// Agent-scoped memory
AddOptions::for_agent("assistant-v1")

// Combined scoping
AddOptions {
    user_id: Some("alice".to_string()),
    agent_id: Some("assistant-v1".to_string()),
    ..Default::default()
}
```

## Examples

Run the examples:

```bash
# Basic usage (in-memory, mock embeddings)
cargo run --example basic_usage

# With OpenAI
cargo run --example async_openai --features openai

# With Ollama (local)
cargo run --example ollama_local --features ollama

# With Qdrant
cargo run --example qdrant_store --features qdrant
```

## Architecture

```
mem0-rust/
├── src/
│   ├── embeddings/      # Embedding providers (Mock, OpenAI, Ollama)
│   ├── vector_stores/   # Vector backends (Memory, Qdrant, Postgres, Redis)
│   ├── llms/            # LLM providers (OpenAI, Ollama, Anthropic)
│   ├── memory/          # Core memory management
│   └── utils/           # Filter builders, utilities
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
