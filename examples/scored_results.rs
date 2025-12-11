use mem0_rust::memory::MemoryClient;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = MemoryClient::default();

    client.add(
        "Rust empowers fearless concurrency and memory safety",
        json!({"topic": "rust", "kind": "statement"}),
    );
    client.add(
        "Borrow checker errors can be intimidating for newcomers",
        json!({"topic": "rust", "kind": "pain_point"}),
    );
    client.add(
        "Pattern matching makes Rust enums ergonomic to use",
        json!({"topic": "rust", "kind": "benefit"}),
    );

    let results = client.search("what helps enums in rust?")?;

    println!("Top results with scores (>= threshold):");
    for result in results {
        println!(
            "- {:.3} | {} | {}",
            result.score,
            result.record.content,
            result.record.created_at.to_rfc3339(),
        );
    }

    Ok(())
}
