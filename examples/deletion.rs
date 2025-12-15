//! Deletion example for mem0-rust.

use mem0_rust::{AddOptions, Memory, MemoryConfig, SearchOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MemoryConfig::default();
    let memory = Memory::new(config).await?;

    // Add a memory
    let result = memory
        .add(
            "Temporary note to delete",
            AddOptions::for_user("user1").raw(),
        )
        .await?;

    let memory_id = result.results[0].id.to_string();
    println!("Added memory with ID: {}", memory_id);

    // Delete the memory
    memory.delete(&memory_id).await?;
    println!("Deleted memory: {}", memory_id);

    // Verify deletion
    let search_results = memory
        .search(
            "Temporary note",
            SearchOptions::for_user("user1"),
        )
        .await?;

    println!("Search after deletion returned {} results", search_results.results.len());

    Ok(())
}
