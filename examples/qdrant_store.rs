//! Qdrant vector store example for mem0-rust.
//!
//! This example requires Qdrant running at localhost:6334.
//! Run: docker run -p 6334:6334 qdrant/qdrant

use mem0_rust::{AddOptions, Memory, MemoryConfig, SearchOptions, VectorStoreConfig};

#[cfg(feature = "qdrant")]
use mem0_rust::config::QdrantConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "qdrant"))]
    {
        eprintln!("This example requires the 'qdrant' feature.");
        eprintln!("Run with: cargo run --example qdrant_store --features qdrant");
        return Ok(());
    }

    #[cfg(feature = "qdrant")]
    {
        println!("Connecting to Qdrant at localhost:6334...");

        // Configure with Qdrant vector store
        let config = MemoryConfig {
            vector_store: VectorStoreConfig::Qdrant(QdrantConfig {
                url: "http://localhost:6334".to_string(),
                collection_name: "mem0_example".to_string(),
                dimensions: 128, // Using mock embedder, so small dimension
                ..Default::default()
            }),
            ..Default::default()
        };

        let memory = Memory::new(config).await?;

        // Add some memories (these will persist in Qdrant!)
        println!("Adding memories to Qdrant...");

        memory
            .add(
                "Qdrant is a vector similarity search engine",
                AddOptions::for_user("admin").raw(),
            )
            .await?;

        memory
            .add(
                "Vector databases enable semantic search",
                AddOptions::for_user("admin").raw(),
            )
            .await?;

        memory
            .add(
                "RAG applications rely on efficient vector retrieval",
                AddOptions::for_user("admin").raw(),
            )
            .await?;

        // Search
        println!("\nSearching in Qdrant...");
        let results = memory
            .search(
                "vector search technology",
                SearchOptions::for_user("admin").with_limit(5),
            )
            .await?;

        println!("Found {} results:", results.results.len());
        for r in &results.results {
            println!("  - {} (score: {:.3})", r.record.content, r.score);
        }

        // Get all memories
        println!("\nAll memories in collection:");
        let all_memories = memory.get_all(mem0_rust::GetAllOptions {
            user_id: Some("admin".to_string()),
            ..Default::default()
        }).await?;

        for m in &all_memories {
            println!("  - [{}] {}", m.id, m.content);
        }

        Ok(())
    }
}
