//! A lightweight Rust reimplementation of the core `mem0` memory client.
//! It provides deterministic embeddings, an in-memory vector store, and a small
//! API for adding, searching, and deleting memories.

pub mod config;
pub mod embedder;
pub mod errors;
pub mod memory;
pub mod models;
pub mod store;

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::config::MemoryConfig;
    use crate::memory::MemoryClient;

    #[test]
    fn add_and_search_memory() {
        let mut client = MemoryClient::default();
        client.add(
            "Rust is great for systems programming",
            json!({"topic": "rust"}),
        );
        client.add(
            "Python is often used for rapid prototyping",
            json!({"topic": "python"}),
        );

        let results = client
            .search("systems programming")
            .expect("search to succeed");

        assert!(!results.is_empty());
        assert_eq!(results[0].record.metadata["topic"], "rust");
    }

    #[test]
    fn delete_memory() {
        let mut client = MemoryClient::default();
        let memory = client.add("Delete me", json!({"id": 1}));
        assert!(client.delete(&memory.id.to_string()).is_ok());
        assert!(client.search("Delete me").unwrap().is_empty());
    }

    #[test]
    fn configurable_limits() {
        let config = MemoryConfig {
            embedding_dim: 64,
            max_results: 1,
            similarity_threshold: 0.0,
        };
        let mut client = MemoryClient::new(config);
        client.add("First item", json!({"order": 1}));
        client.add("Second item", json!({"order": 2}));

        let results = client.search("item").expect("search to work");
        assert_eq!(results.len(), 1);
    }
}
