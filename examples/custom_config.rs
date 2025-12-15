//! Custom configuration example for mem0-rust.

use mem0_rust::{AddOptions, Memory, MemoryConfig, MockEmbedderConfig, EmbedderConfig, SearchOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a custom configuration
    let config = MemoryConfig {
        embedder: EmbedderConfig::Mock(MockEmbedderConfig {
            dimensions: 256, // Use larger dimensions
        }),
        collection_name: "custom_memories".to_string(),
        ..Default::default()
    };

    let memory = Memory::new(config).await?;

    // Add memories
    memory
        .add(
            "The Eiffel Tower is in Paris",
            AddOptions::for_user("tourist").raw(),
        )
        .await?;

    memory
        .add(
            "The Great Wall is in China",
            AddOptions::for_user("tourist").raw(),
        )
        .await?;

    // Search with threshold
    let results = memory
        .search(
            "famous landmarks in France",
            SearchOptions::for_user("tourist")
                .with_limit(5)
                .with_threshold(0.1),
        )
        .await?;

    println!("Landmarks matching 'France':");
    for result in &results.results {
        println!("- {} (score: {:.3})", result.record.content, result.score);
    }

    Ok(())
}
