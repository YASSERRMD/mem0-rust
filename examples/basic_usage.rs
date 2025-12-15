//! Basic usage example for mem0-rust.
//!
//! This example demonstrates how to create a Memory instance,
//! add memories, and search for them.

use mem0_rust::{AddOptions, Memory, MemoryConfig, SearchOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a memory instance with default config (in-memory store, mock embedder)
    let config = MemoryConfig::default();
    let memory = Memory::new(config).await?;

    // Add some memories for a user
    let user_id = "alice";

    memory
        .add(
            "I love programming in Rust",
            AddOptions {
                user_id: Some(user_id.to_string()),
                infer: false, // Disable LLM inference for this example
                ..Default::default()
            },
        )
        .await?;

    memory
        .add(
            "My favorite food is pizza",
            AddOptions {
                user_id: Some(user_id.to_string()),
                infer: false,
                ..Default::default()
            },
        )
        .await?;

    memory
        .add(
            "I work as a software engineer",
            AddOptions {
                user_id: Some(user_id.to_string()),
                infer: false,
                ..Default::default()
            },
        )
        .await?;

    // Search for relevant memories
    let results = memory
        .search(
            "programming languages",
            SearchOptions {
                user_id: Some(user_id.to_string()),
                limit: Some(3),
                ..Default::default()
            },
        )
        .await?;

    println!("Top results:");
    for result in &results.results {
        println!(
            "- {} (score: {:.3})",
            result.record.content, result.score
        );
    }

    Ok(())
}
