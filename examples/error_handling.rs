use mem0_rust::config::MemoryConfig;
use mem0_rust::memory::MemoryClient;
use serde_json::json;

fn main() {
    // Use a small result set and permissive similarity threshold.
    let config = MemoryConfig {
        embedding_dim: 16,
        similarity_threshold: 0.0,
        max_results: 2,
    };
    let mut client = MemoryClient::new(config);

    let record = client.add("Offline-first apps cache writes", json!({"tag": "architecture"}));
    println!("Stored record {}", record.id);

    // Safe deletion with explicit error handling if the id is unknown.
    match client.delete("non-existent-id") {
        Ok(_) => println!("Unexpectedly deleted a record"),
        Err(err) => println!("Delete failed as expected: {err}"),
    }

    // Search with the limited result set to illustrate max_results behavior.
    client.add("CRDTs reconcile concurrent updates", json!({"tag": "distributed"}));
    client.add("Vector clocks track causality", json!({"tag": "distributed"}));

    let results = client.search("offline updates").expect("search to succeed");
    println!("Received {} results (bounded by max_results)", results.len());
    for item in results {
        println!("- {} [score {:.3}]", item.record.content, item.score);
    }
}
