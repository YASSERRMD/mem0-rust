//! Ollama local models example for mem0-rust.
//!
//! This example requires Ollama running locally with the nomic-embed-text model.
//! Run: ollama pull nomic-embed-text

use mem0_rust::{AddOptions, EmbedderConfig, Memory, MemoryConfig, SearchOptions};

#[cfg(feature = "ollama")]
use mem0_rust::config::OllamaEmbedderConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "ollama"))]
    {
        eprintln!("This example requires the 'ollama' feature.");
        eprintln!("Run with: cargo run --example ollama_local --features ollama");
        return Ok(());
    }

    #[cfg(feature = "ollama")]
    {
        println!("Connecting to Ollama at localhost:11434...");

        // Configure with Ollama embeddings
        let config = MemoryConfig {
            embedder: EmbedderConfig::Ollama(OllamaEmbedderConfig {
                model: "nomic-embed-text".to_string(),
                base_url: "http://localhost:11434".to_string(),
                dimensions: 768,
            }),
            ..Default::default()
        };

        let memory = Memory::new(config).await?;

        // Add some memories
        println!("Adding memories...");

        memory
            .add(
                "Local LLMs provide privacy and control over your data",
                AddOptions::for_user("developer").raw(),
            )
            .await?;

        memory
            .add(
                "Ollama makes it easy to run open-source models",
                AddOptions::for_user("developer").raw(),
            )
            .await?;

        memory
            .add(
                "Self-hosted AI can reduce latency and costs",
                AddOptions::for_user("developer").raw(),
            )
            .await?;

        // Search
        println!("\nSearching for 'benefits of running AI locally'...");
        let results = memory
            .search(
                "benefits of running AI locally",
                SearchOptions::for_user("developer").with_limit(5),
            )
            .await?;

        println!("Found {} results:", results.results.len());
        for r in &results.results {
            println!("  - {} (score: {:.3})", r.record.content, r.score);
        }

        Ok(())
    }
}
