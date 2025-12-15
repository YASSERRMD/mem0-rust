//! OpenAI integration example for mem0-rust.
//!
//! This example requires the `openai` feature and OPENAI_API_KEY env var.

use mem0_rust::{
    AddOptions, EmbedderConfig, LLMConfig, Memory, MemoryConfig, SearchOptions,
};

#[cfg(feature = "openai")]
use mem0_rust::config::{OpenAIEmbedderConfig, OpenAILLMConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "openai"))]
    {
        eprintln!("This example requires the 'openai' feature.");
        eprintln!("Run with: cargo run --example async_openai --features openai");
        return Ok(());
    }

    #[cfg(feature = "openai")]
    {
        // Check for API key
        if std::env::var("OPENAI_API_KEY").is_err() {
            eprintln!("Please set OPENAI_API_KEY environment variable");
            return Ok(());
        }

        // Configure with OpenAI embeddings and LLM
        let config = MemoryConfig {
            embedder: EmbedderConfig::OpenAI(OpenAIEmbedderConfig {
                model: "text-embedding-3-small".to_string(),
                dimensions: Some(1536),
                ..Default::default()
            }),
            llm: Some(LLMConfig::OpenAI(OpenAILLMConfig {
                model: "gpt-4o-mini".to_string(),
                temperature: 0.0,
                ..Default::default()
            })),
            ..Default::default()
        };

        let memory = Memory::new(config).await?;

        // Add a conversation with LLM inference
        println!("Adding memories with LLM inference...");
        
        let result = memory
            .add(
                vec![
                    mem0_rust::Message::user("Hi, I'm John. I work as a data scientist at Google."),
                    mem0_rust::Message::assistant("Nice to meet you, John! That sounds like an exciting role."),
                    mem0_rust::Message::user("Yes! I specialize in NLP and love working with transformers."),
                ],
                AddOptions {
                    user_id: Some("john".to_string()),
                    infer: true, // Enable LLM inference
                    ..Default::default()
                },
            )
            .await?;

        println!("Extracted {} memories:", result.results.len());
        for r in &result.results {
            println!("  - {} ({})", r.memory, r.event.to_string());
        }

        // Search for relevant memories
        println!("\nSearching for 'machine learning work'...");
        let search_results = memory
            .search(
                "machine learning work",
                SearchOptions::for_user("john").with_limit(5),
            )
            .await?;

        println!("Found {} results:", search_results.results.len());
        for r in &search_results.results {
            println!("  - {} (score: {:.3})", r.record.content, r.score);
        }

        Ok(())
    }
}
