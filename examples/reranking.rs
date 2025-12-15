//! Reranking example using Cohere through mem0-rust.

use mem0_rust::{AddOptions, CohereRerankerConfig, Memory, MemoryConfig, RerankerConfig, SearchOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("COHERE_API_KEY").is_err() {
        eprintln!("Please set COHERE_API_KEY environment variable");
        return Ok(());
    }

    println!("Using Cohere Reranker...");

    let config = MemoryConfig {
        reranker: Some(RerankerConfig::Cohere(CohereRerankerConfig {
            model: "rerank-english-v3.0".to_string(),
            ..Default::default()
        })),
        ..Default::default()
    };

    let memory = Memory::new(config).await?;

    println!("Adding memories...");
    let docs = vec![
        "The capital of France is Paris",
        "France has a population of 67 million",
        "Paris is known for the Eiffel Tower",
        "Berlin is the capital of Germany",
        "Rome is the capital of Italy",
    ];

    for doc in docs {
        memory.add(doc, AddOptions::for_user("student").raw()).await?;
    }

    let query = "What is the capital of France?";
    
    println!("\nSearching for '{}' with reranking...", query);
    let results = memory.search(query, SearchOptions {
        user_id: Some("student".to_string()), 
        rerank: true, 
        limit: Some(3),
        ..Default::default()
    }).await?;

    println!("Top 3 results:");
    for r in results.results {
        println!("- {} (score: {:.3})", r.record.content, r.score);
    }

    Ok(())
}
