//! Error handling example for mem0-rust.

use mem0_rust::{AddOptions, Memory, MemoryConfig, MemoryError, SearchOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MemoryConfig::default();
    let memory = Memory::new(config).await?;

    // Add a valid memory
    let result = memory
        .add(
            "Valid memory content",
            AddOptions::for_user("user1").raw(),
        )
        .await?;
    println!("Added memory: {}", result.results[0].id);

    // Try to delete a non-existent memory
    match memory.delete("non-existent-id").await {
        Ok(_) => println!("Deleted successfully"),
        Err(MemoryError::NotFound(id)) => {
            println!("Error: Memory with id '{}' not found", id);
        }
        Err(e) => {
            println!("Unexpected error: {}", e);
        }
    }

    // Try to add without scoping (will fail)
    match memory
        .add(
            "Memory without user_id",
            AddOptions {
                infer: false,
                ..Default::default()
            },
        )
        .await
    {
        Ok(_) => println!("Added successfully"),
        Err(MemoryError::InvalidInput(msg)) => {
            println!("Validation error: {}", msg);
        }
        Err(e) => {
            println!("Unexpected error: {}", e);
        }
    }

    // Search with limit
    memory
        .add("CRDTs reconcile concurrent updates", AddOptions::for_user("dev").raw())
        .await?;
    memory
        .add("Vector clocks track causality", AddOptions::for_user("dev").raw())
        .await?;
    memory
        .add("Consensus algorithms ensure agreement", AddOptions::for_user("dev").raw())
        .await?;

    let results = memory
        .search(
            "distributed systems",
            SearchOptions::for_user("dev").with_limit(2),
        )
        .await?;

    println!("\nReceived {} results (bounded by limit)", results.results.len());
    for r in &results.results {
        println!("- {} [score {:.3}]", r.record.content, r.score);
    }

    Ok(())
}
