use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use crate::errors::MemoryError;
use crate::models::ScoredMemory;
use crate::config::CohereRerankerConfig;
use super::Reranker;

pub struct CohereReranker {
    client: Client,
    api_key: String,
    model: String,
}

impl CohereReranker {
    pub fn new(config: CohereRerankerConfig) -> Result<Self, MemoryError> {
        let api_key = config.api_key
             .or_else(|| std::env::var("COHERE_API_KEY").ok())
             .ok_or_else(|| MemoryError::Config("COHERE_API_KEY not set".to_string()))?;
             
        Ok(Self {
            client: Client::new(),
            api_key,
            model: config.model,
        })
    }
}

#[derive(Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: Vec<&'a str>,
    top_n: usize,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    relevance_score: f32,
}

#[async_trait]
impl Reranker for CohereReranker {
    async fn rerank(&self, query: &str, results: Vec<ScoredMemory>) -> Result<Vec<ScoredMemory>, MemoryError> {
        if results.is_empty() {
            return Ok(results);
        }

        let documents: Vec<&str> = results.iter().map(|m| m.record.content.as_str()).collect();

        let request = RerankRequest {
            model: &self.model,
            query,
            documents,
            top_n: results.len(),
        };

        let response = self.client.post("https://api.cohere.com/v1/rerank")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| MemoryError::Reranker(e.to_string()))?;

        if !response.status().is_success() {
             let error_text = response.text().await.unwrap_or_default();
             return Err(MemoryError::Reranker(format!("Cohere API error: {}", error_text)));
        }

        let rerank_response: RerankResponse = response.json().await
            .map_err(|e| MemoryError::Reranker(format!("Failed to parse response: {}", e)))?;

        let mut reranked = Vec::new();
        for result in rerank_response.results {
            if let Some(mut memory) = results.get(result.index).cloned() {
                memory.score = result.relevance_score;
                reranked.push(memory);
            }
        }

        Ok(reranked)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
