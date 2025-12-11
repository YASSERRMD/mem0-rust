use mem0_rust::memory::MemoryClient;
use serde_json::json;

fn main() {
    let mut client = MemoryClient::default();

    // Store conversation turns with speaker metadata.
    client.add(
        "User: How do I cook risotto?",
        json!({"speaker": "user", "topic": "cooking"}),
    );
    client.add(
        "Assistant: Sauté onions, toast rice, and add broth gradually.",
        json!({"speaker": "assistant", "topic": "cooking"}),
    );
    client.add(
        "Assistant: Rust ownership prevents data races.",
        json!({"speaker": "assistant", "topic": "rust"}),
    );

    let results = client
        .search("make creamy rice")
        .expect("search to succeed");

    println!("Relevant answers from the assistant:");
    for item in results
        .iter()
        .filter(|r| r.record.metadata.get("speaker") == Some(&json!("assistant")))
    {
        println!("- {} (topic: {})", item.record.content, item.record.metadata["topic"]);
    }
}
