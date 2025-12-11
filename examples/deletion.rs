use mem0_rust::memory::MemoryClient;
use serde_json::json;

fn main() {
    let mut client = MemoryClient::default();

    let record = client.add("Temporary note", json!({"category": "temp"}));
    println!("Stored memory {}: {}", record.id, record.content);

    client
        .delete(&record.id.to_string())
        .expect("delete should succeed");
    println!("Deleted record {}", record.id);

    let results = client.search("Temporary note").expect("search to work");
    println!("Search after deletion returned {} results", results.len());
}
