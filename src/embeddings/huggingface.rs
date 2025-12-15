//! HuggingFace Inference API embeddings provider.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::traits::Embedder;
use crate::config::HuggingFaceEmbedderConfig;
use crate::errors::EmbeddingError;

/// HuggingFace embeddings provider
pub struct HuggingFaceEmbedder {
    client: Client,
    api_key: String,
    model: String,
    dimensions: usize,
    api_url: String,
}

impl HuggingFaceEmbedder {
    /// Create a new HuggingFace embedder
    pub fn new(config: HuggingFaceEmbedderConfig) -> Result<Self, EmbeddingError> {
        let api_key = config
            .api_key
            .or_else(|| std::env::var("HF_TOKEN").ok())
            .ok_or_else(|| EmbeddingError::Api("HF_TOKEN not set".to_string()))?;

        let api_url = config.api_url.unwrap_or_else(|| {
            format!(
                "https://api-inference.huggingface.co/pipeline/feature-extraction/{}",
                config.model
            )
        });

        Ok(Self {
            client: Client::new(),
            api_key,
            model: config.model,
            dimensions: config.dimensions,
            api_url,
        })
    }
}

#[derive(Debug, Serialize)]
struct HFRequest {
    inputs: Vec<String>,
    options: HFOptions,
}

#[derive(Debug, Serialize)]
struct HFOptions {
    wait_for_model: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum HFResponse {
    Single(Vec<f32>),
    Batch(Vec<Vec<f32>>),
    // Some models return nested arrays (token-level embeddings)
    Nested(Vec<Vec<Vec<f32>>>),
}

#[async_trait]
impl Embedder for HuggingFaceEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let request = HFRequest {
            inputs: vec![text.to_string()],
            options: HFOptions {
                wait_for_model: true,
            },
        };

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::Api(format!(
                "HuggingFace API error: {}",
                error_text
            )));
        }

        let result: HFResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::InvalidResponse(e.to_string()))?;

        match result {
            HFResponse::Single(embedding) => Ok(embedding),
            HFResponse::Batch(embeddings) => embeddings
                .into_iter()
                .next()
                .ok_or_else(|| EmbeddingError::InvalidResponse("Empty response".to_string())),
            HFResponse::Nested(nested) => {
                // Some models return [[embedding]] format - apply mean pooling
                nested
                    .into_iter()
                    .next()
                    .and_then(|inner| {
                        if inner.is_empty() {
                            return None;
                        }
                        let dim = inner[0].len();
                        let mut pooled = vec![0.0f32; dim];
                        for token_emb in &inner {
                            for (i, v) in token_emb.iter().enumerate() {
                                if i < dim {
                                    pooled[i] += v;
                                }
                            }
                        }
                        let n = inner.len() as f32;
                        for v in &mut pooled {
                            *v /= n;
                        }
                        Some(pooled)
                    })
                    .ok_or_else(|| {
                        EmbeddingError::InvalidResponse("Empty nested response".to_string())
                    })
            }
        }
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let request = HFRequest {
            inputs: texts.iter().map(|s| s.to_string()).collect(),
            options: HFOptions {
                wait_for_model: true,
            },
        };

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::Api(format!(
                "HuggingFace API error: {}",
                error_text
            )));
        }

        let result: HFResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::InvalidResponse(e.to_string()))?;

        match result {
            HFResponse::Single(embedding) => Ok(vec![embedding]),
            HFResponse::Batch(embeddings) => Ok(embeddings),
            HFResponse::Nested(nested) => {
                // Mean pooling for each text
                nested
                    .into_iter()
                    .map(|inner| {
                        if inner.is_empty() {
                            return Err(EmbeddingError::InvalidResponse(
                                "Empty nested response".to_string(),
                            ));
                        }
                        let dim = inner[0].len();
                        let mut pooled = vec![0.0f32; dim];
                        for token_emb in &inner {
                            for (i, v) in token_emb.iter().enumerate() {
                                if i < dim {
                                    pooled[i] += v;
                                }
                            }
                        }
                        let n = inner.len() as f32;
                        for v in &mut pooled {
                            *v /= n;
                        }
                        Ok(pooled)
                    })
                    .collect()
            }
        }
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
