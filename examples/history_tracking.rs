//! History tracking example for mem0-rust.

use mem0_rust::{AddOptions, Memory, MemoryConfig};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure with history DB
    let history_db = PathBuf::from("history.db");
    
    // Clean up previous run
    if history_db.exists() {
        std::fs::remove_file(&history_db)?;
    }

    let config = MemoryConfig {
        history_db_path: Some(history_db.clone()),
        ..Default::default()
    };

    let memory = Memory::new(config).await?;

    // Add memory
    println!("Adding memory...");
    let result = memory
        .add(
            "Initial memory content",
            AddOptions::for_user("alice").raw(),
        )
        .await?;
    
    let id = result.results[0].id.to_string();
    println!("Added memory ID: {}", id);

    // Update memory
    println!("Updating memory...");
    memory.update(&id, "Updated memory content").await?;

    // Get history
    let history = memory.history(&id).await?;
    
    println!("\nHistory for {}:", id);
    for entry in history {
        println!("- [{}] {:?} (New: '{}', Old: '{:?}')", 
            entry.timestamp,
            entry.event,
            entry.new_content,
            entry.previous_content
        );
    }

    // Cleanup
    let _ = std::fs::remove_file(history_db);

    Ok(())
}
