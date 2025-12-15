//! Metadata filtering example for mem0-rust.

use mem0_rust::{AddOptions, Memory, MemoryConfig, SearchOptions};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MemoryConfig::default();
    let memory = Memory::new(config).await?;

    // Add memories with different metadata
    let mut metadata = HashMap::new();
    metadata.insert("role".to_string(), serde_json::json!("user"));
    metadata.insert("category".to_string(), serde_json::json!("question"));

    memory
        .add(
            "What is the weather like today?",
            AddOptions {
                user_id: Some("user1".to_string()),
                metadata: Some(metadata.clone()),
                infer: false,
                ..Default::default()
            },
        )
        .await?;

    metadata.insert("role".to_string(), serde_json::json!("assistant"));
    metadata.insert("category".to_string(), serde_json::json!("answer"));

    memory
        .add(
            "The weather is sunny and warm, around 25 degrees.",
            AddOptions {
                user_id: Some("user1".to_string()),
                metadata: Some(metadata),
                infer: false,
                ..Default::default()
            },
        )
        .await?;

    // Search for answers
    let results = memory
        .search(
            "weather information",
            SearchOptions::for_user("user1").with_limit(5),
        )
        .await?;

    println!("Weather-related memories:");
    for result in &results.results {
        let role = result.record.metadata.get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        println!("- [{}] {}", role, result.record.content);
    }

    Ok(())
}
