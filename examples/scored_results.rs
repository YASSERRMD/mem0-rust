//! Scored results example for mem0-rust.
//!
//! Demonstrates how to use similarity thresholds and understand scores.

use mem0_rust::{AddOptions, Memory, MemoryConfig, SearchOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MemoryConfig::default();
    let memory = Memory::new(config).await?;

    // Add some programming-related memories
    let memories = vec![
        "Rust ownership prevents data races at compile time",
        "Pattern matching makes Rust enums ergonomic to use",
        "Borrow checker errors can be intimidating for newcomers",
        "WebAssembly allows Rust to run in browsers",
        "async/await simplifies asynchronous programming in Rust",
    ];

    for content in memories {
        memory
            .add(content, AddOptions::for_user("dev").raw())
            .await?;
    }

    // Search with different thresholds
    println!("=== Search: 'memory safety in Rust' ===\n");

    let results = memory
        .search(
            "memory safety in Rust",
            SearchOptions::for_user("dev")
                .with_limit(10)
                .with_threshold(0.0), // Get all results
        )
        .await?;

    println!("All results (no threshold):");
    for r in &results.results {
        println!("  {:.3} | {}", r.score, r.record.content);
    }

    let high_threshold = memory
        .search(
            "memory safety in Rust",
            SearchOptions::for_user("dev")
                .with_limit(10)
                .with_threshold(0.3), // Only high scores
        )
        .await?;

    println!("\nHigh threshold (>= 0.3):");
    for r in &high_threshold.results {
        println!("  {:.3} | {}", r.score, r.record.content);
    }

    Ok(())
}
