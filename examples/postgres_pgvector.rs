//! PostgreSQL with pgvector example for mem0-rust.
//!
//! This example requires PostgreSQL with pgvector extension.
//! Run: docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres ankane/pgvector

use mem0_rust::{AddOptions, Memory, MemoryConfig, SearchOptions, VectorStoreConfig};

#[cfg(feature = "postgres")]
use mem0_rust::config::PostgresConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "postgres"))]
    {
        eprintln!("This example requires the 'postgres' feature.");
        eprintln!("Run with: cargo run --example postgres_pgvector --features postgres");
        return Ok(());
    }

    #[cfg(feature = "postgres")]
    {
        println!("Connecting to PostgreSQL at localhost:5432...");

        // Configure with PostgreSQL + pgvector
        let config = MemoryConfig {
            vector_store: VectorStoreConfig::Postgres(PostgresConfig {
                connection_url: "postgres://postgres:postgres@localhost:5432/postgres".to_string(),
                table_name: "memories".to_string(),
                dimensions: 128, // Using mock embedder
            }),
            ..Default::default()
        };

        let memory = Memory::new(config).await?;

        // Add some memories
        println!("Adding memories to PostgreSQL...");

        memory
            .add(
                "PostgreSQL is a powerful open-source database",
                AddOptions::for_user("dba").raw(),
            )
            .await?;

        memory
            .add(
                "pgvector enables vector similarity search in PostgreSQL",
                AddOptions::for_user("dba").raw(),
            )
            .await?;

        memory
            .add(
                "ACID compliance ensures data integrity",
                AddOptions::for_user("dba").raw(),
            )
            .await?;

        // Search
        println!("\nSearching in PostgreSQL...");
        let results = memory
            .search(
                "vector database features",
                SearchOptions::for_user("dba").with_limit(5),
            )
            .await?;

        println!("Found {} results:", results.results.len());
        for r in &results.results {
            println!("  - {} (score: {:.3})", r.record.content, r.score);
        }

        Ok(())
    }
}
