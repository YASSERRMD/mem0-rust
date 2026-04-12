#![cfg(feature = "server")]

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use mem0_rust::{AddOptions, Memory, MemoryConfig, SearchOptions};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
struct AppState {
    memory: Arc<Memory>,
}

#[derive(Debug, Deserialize)]
struct AddRequest {
    text: String,
    user_id: String,
}

#[derive(Debug, Serialize)]
struct AddResponse {
    events: usize,
}

#[derive(Debug, Deserialize)]
struct SearchRequest {
    query: String,
    user_id: String,
    limit: Option<usize>,
}

#[derive(Debug, Serialize)]
struct SearchItem {
    score: f32,
    content: String,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    results: Vec<SearchItem>,
}

async fn health() -> &'static str {
    "ok"
}

async fn add_memory(
    State(state): State<AppState>,
    Json(req): Json<AddRequest>,
) -> Result<Json<AddResponse>, String> {
    let added = state
        .memory
        .add(
            req.text,
            AddOptions {
                user_id: Some(req.user_id),
                ..Default::default()
            },
        )
        .await
        .map_err(|e| e.to_string())?;

    Ok(Json(AddResponse {
        events: added.results.len(),
    }))
}

async fn search_memory(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, String> {
    let found = state
        .memory
        .search(
            &req.query,
            SearchOptions {
                user_id: Some(req.user_id),
                limit: req.limit.or(Some(5)),
                ..Default::default()
            },
        )
        .await
        .map_err(|e| e.to_string())?;

    let results = found
        .results
        .into_iter()
        .map(|row| SearchItem {
            score: row.score,
            content: row.record.content,
        })
        .collect();

    Ok(Json(SearchResponse { results }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let memory = Arc::new(Memory::new(MemoryConfig::default()).await?);
    let state = AppState { memory };

    let app = Router::new()
        .route("/health", get(health))
        .route("/add", post(add_memory))
        .route("/search", post(search_memory))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("mem0 server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
