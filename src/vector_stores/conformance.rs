#![cfg(test)]

use std::collections::HashMap;

use chrono::Utc;

use super::VectorStore;
use crate::models::Payload;

fn payload(data: &str, category: &str) -> Payload {
    let mut metadata = HashMap::new();
    metadata.insert("category".to_string(), serde_json::json!(category));

    Payload {
        data: data.to_string(),
        hash: "test_hash".to_string(),
        created_at: Utc::now(),
        user_id: None,
        agent_id: None,
        run_id: None,
        metadata,
    }
}

pub async fn run_basic_contract<T: VectorStore>(store: &T) {
    store.create_collection().await.unwrap();
    assert!(store.collection_exists().await.unwrap());

    store
        .insert("id-1", vec![1.0, 0.0], payload("alpha", "a"))
        .await
        .unwrap();
    store
        .insert("id-2", vec![0.0, 1.0], payload("beta", "b"))
        .await
        .unwrap();

    let search = store.search(&[1.0, 0.0], 1, None).await.unwrap();
    assert_eq!(search.len(), 1);
    assert_eq!(search[0].id, "id-1");

    let fetched = store.get("id-1").await.unwrap();
    assert!(fetched.is_some());

    store
        .update("id-2", Some(vec![1.0, 0.0]), payload("beta-2", "b"))
        .await
        .unwrap();

    let updated = store.search(&[1.0, 0.0], 2, None).await.unwrap();
    assert_eq!(updated.len(), 2);

    let all = store.list(None, 10).await.unwrap();
    assert_eq!(all.len(), 2);

    let deleted = store.delete_all(None).await.unwrap();
    assert_eq!(deleted, 2);

    let empty = store.list(None, 10).await.unwrap();
    assert!(empty.is_empty());
}
