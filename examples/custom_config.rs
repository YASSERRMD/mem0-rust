use mem0_rust::config::MemoryConfig;
use mem0_rust::memory::MemoryClient;
use serde_json::json;

fn main() {
    // Tune embedding dimension, similarity threshold, and max results.
    let config = MemoryConfig {
        embedding_dim: 32,
        similarity_threshold: 0.2,
        max_results: 3,
    };

    let mut client = MemoryClient::new(config);

    client.add(
        "The Eiffel Tower is in Paris",
        json!({"tag": "geography"}),
    );
    client.add(
        "The Louvre houses the Mona Lisa",
        json!({"tag": "museum"}),
    );
    client.add(
        "Rust's ownership model guarantees memory safety",
        json!({"tag": "programming"}),
    );

    let results = client.search("Paris museum").expect("search to work");

    println!("Limited results with a similarity threshold:");
    for (index, item) in results.iter().enumerate() {
        println!("{}. {} (score {:.3})", index + 1, item.record.content, item.score);
    }
}
