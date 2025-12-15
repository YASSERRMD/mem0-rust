//! HuggingFace embeddings example for mem0-rust.
//!
//! This example uses HuggingFace Inference API for embeddings.
//! Set HF_TOKEN environment variable with your HuggingFace token.

use mem0_rust::{
    AddOptions, EmbedderConfig, Memory, MemoryConfig, SearchOptions,
    config::HuggingFaceEmbedderConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for API token
    if std::env::var("HF_TOKEN").is_err() {
        eprintln!("Please set HF_TOKEN environment variable");
        eprintln!("Get a token at: https://huggingface.co/settings/tokens");
        return Ok(());
    }

    println!("Using HuggingFace Inference API...");

    // Configure with HuggingFace embeddings
    let config = MemoryConfig {
        embedder: EmbedderConfig::HuggingFace(HuggingFaceEmbedderConfig {
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            dimensions: 384,
            ..Default::default()
        }),
        ..Default::default()
    };

    let memory = Memory::new(config).await?;

    // Add some memories
    println!("Adding memories...");

    memory
        .add(
            "Machine learning models can recognize patterns in data",
            AddOptions::for_user("researcher").raw(),
        )
        .await?;

    memory
        .add(
            "Deep learning uses neural networks with many layers",
            AddOptions::for_user("researcher").raw(),
        )
        .await?;

    memory
        .add(
            "Transformers have revolutionized natural language processing",
            AddOptions::for_user("researcher").raw(),
        )
        .await?;

    // Search using semantic similarity
    println!("\nSearching for 'AI and neural networks'...");
    let results = memory
        .search(
            "AI and neural networks",
            SearchOptions::for_user("researcher").with_limit(5),
        )
        .await?;

    println!("Found {} results:", results.results.len());
    for r in &results.results {
        println!("  - {} (score: {:.3})", r.record.content, r.score);
    }

    Ok(())
}
