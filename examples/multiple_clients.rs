use mem0_rust::config::MemoryConfig;
use mem0_rust::memory::MemoryClient;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A precise client tuned for stricter similarity and fewer hits.
    let strict_config = MemoryConfig {
        embedding_dim: 64,
        similarity_threshold: 0.6,
        max_results: 2,
    };
    let mut strict_client = MemoryClient::new(strict_config);

    // A broad client that returns more candidates for downstream filtering.
    let broad_config = MemoryConfig {
        embedding_dim: 16,
        similarity_threshold: 0.1,
        max_results: 5,
    };
    let mut broad_client = MemoryClient::new(broad_config);

    let snippets = [
        "GraphQL APIs benefit from schema stitching",
        "REST endpoints are easy to cache at CDNs",
        "gRPC shines when you need bidirectional streaming",
        "GraphQL clients can overfetch when queries are poorly designed",
    ];

    for text in snippets {
        strict_client.add(text, json!({"category": "api"}));
        broad_client.add(text, json!({"category": "api"}));
    }

    let query = "streaming api design";

    let strict_results = strict_client.search(query)?;
    println!("Strict client returned {} result(s):", strict_results.len());
    for result in &strict_results {
        println!("- {:.3} | {}", result.score, result.record.content);
    }

    let broad_results = broad_client.search(query)?;
    println!("\nBroad client returned {} result(s):", broad_results.len());
    for result in &broad_results {
        println!("- {:.3} | {}", result.score, result.record.content);
    }

    Ok(())
}
