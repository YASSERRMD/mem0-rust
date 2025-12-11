use mem0_rust::memory::MemoryClient;
use serde_json::json;

fn main() {
    // Create a client with default configuration.
    let mut client = MemoryClient::default();

    // Add a couple of memories with attached metadata.
    client.add(
        "Rust makes systems programming approachable",
        json!({"language": "rust", "kind": "blog"}),
    );
    client.add(
        "Python is popular for data science",
        json!({"language": "python", "kind": "article"}),
    );

    // Search for a query and print the top matches.
    let results = client
        .search("systems programming")
        .expect("search to succeed");

    println!("Top results:");
    for item in results {
        println!("- {} (metadata: {})", item.record.content, item.record.metadata);
    }
}
