# mem0-rust

This crate is a Rust-first take on the `mem0` memory client. It mirrors the Python package's
core ergonomics—adding memories, embedding them, searching for the most relevant matches, and
removing entries—but implemented entirely in Rust with deterministic hashing-based embeddings and
an in-memory vector store.

## Features
- Deterministic, dependency-free embeddings via hashing
- In-memory vector store with cosine similarity search
- Simple API for adding, querying, and deleting memories
- Configurable embedding dimension, similarity threshold, and maximum results

## Quick start
```bash
cargo test
```

Use the `MemoryClient` for basic operations:
```rust
use mem0_rust::memory::MemoryClient;
use serde_json::json;

fn main() {
    let mut client = MemoryClient::default();
    client.add("Rust stores your memories safely", json!({"source": "example"}));
    let results = client.search("stores memories").unwrap();
    println!("Found {} results", results.len());
}
```
