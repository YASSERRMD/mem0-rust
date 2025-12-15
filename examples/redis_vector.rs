//! Redis with vector search example for mem0-rust.
//!
//! This example requires Redis Stack (with RediSearch).
//! Run: docker run -p 6379:6379 redis/redis-stack-server:latest

use mem0_rust::{AddOptions, Memory, MemoryConfig, SearchOptions, VectorStoreConfig};

#[cfg(feature = "redis")]
use mem0_rust::config::RedisConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "redis"))]
    {
        eprintln!("This example requires the 'redis' feature.");
        eprintln!("Run with: cargo run --example redis_vector --features redis");
        return Ok(());
    }

    #[cfg(feature = "redis")]
    {
        println!("Connecting to Redis at localhost:6379...");

        // Configure with Redis vector search
        let config = MemoryConfig {
            vector_store: VectorStoreConfig::Redis(RedisConfig {
                url: "redis://localhost:6379".to_string(),
                index_name: "mem0_idx".to_string(),
                dimensions: 128, // Using mock embedder
            }),
            ..Default::default()
        };

        let memory = Memory::new(config).await?;

        // Add some memories
        println!("Adding memories to Redis...");

        memory
            .add(
                "Redis is an in-memory data store",
                AddOptions::for_user("sre").raw(),
            )
            .await?;

        memory
            .add(
                "RediSearch enables full-text and vector search",
                AddOptions::for_user("sre").raw(),
            )
            .await?;

        memory
            .add(
                "Redis Pub/Sub enables real-time messaging",
                AddOptions::for_user("sre").raw(),
            )
            .await?;

        // Search
        println!("\nSearching in Redis...");
        let results = memory
            .search(
                "search capabilities",
                SearchOptions::for_user("sre").with_limit(5),
            )
            .await?;

        println!("Found {} results:", results.results.len());
        for r in &results.results {
            println!("  - {} (score: {:.3})", r.record.content, r.score);
        }

        Ok(())
    }
}
