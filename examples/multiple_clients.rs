//! Multiple clients example for mem0-rust.
//!
//! Demonstrates how different users have isolated memory spaces.

use mem0_rust::{AddOptions, Memory, MemoryConfig, SearchOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MemoryConfig::default();
    let memory = Memory::new(config).await?;

    // User 1 adds their preferences
    memory
        .add(
            "I love hiking in the mountains",
            AddOptions::for_user("alice").raw(),
        )
        .await?;
    
    memory
        .add(
            "My favorite color is blue",
            AddOptions::for_user("alice").raw(),
        )
        .await?;

    // User 2 adds different preferences
    memory
        .add(
            "I prefer beaches over mountains",
            AddOptions::for_user("bob").raw(),
        )
        .await?;
    
    memory
        .add(
            "I like the color green",
            AddOptions::for_user("bob").raw(),
        )
        .await?;

    // Search for Alice's preferences
    let alice_results = memory
        .search(
            "outdoor activities",
            SearchOptions::for_user("alice").with_limit(5),
        )
        .await?;

    println!("Alice's memories about outdoor activities:");
    for r in &alice_results.results {
        println!("- {} (score: {:.3})", r.record.content, r.score);
    }

    // Search for Bob's preferences
    let bob_results = memory
        .search(
            "outdoor activities",
            SearchOptions::for_user("bob").with_limit(5),
        )
        .await?;

    println!("\nBob's memories about outdoor activities:");
    for r in &bob_results.results {
        println!("- {} (score: {:.3})", r.record.content, r.score);
    }

    Ok(())
}
