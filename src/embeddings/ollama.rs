//! Ollama embeddings provider.

use async_trait::async_trait;
use ollama_rs::Ollama;

use super::traits::Embedder;
use crate::config::OllamaEmbedderConfig;
use crate::errors::EmbeddingError;

/// Ollama embeddings provider for local models
pub struct OllamaEmbedder {
    client: Ollama,
    model: String,
    dimensions: usize,
}

impl OllamaEmbedder {
    /// Create a new Ollama embedder
    pub fn new(config: OllamaEmbedderConfig) -> Self {
        let url = url::Url::parse(&config.base_url).unwrap_or_else(|_| {
            url::Url::parse("http://localhost:11434").unwrap()
        });
        
        let host = url.host_str().unwrap_or("localhost").to_string();
        let port = url.port().unwrap_or(11434);

        let client = Ollama::new(host, port);

        Self {
            client,
            model: config.model,
            dimensions: config.dimensions,
        }
    }
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let response = self
            .client
            .generate_embeddings(self.model.clone(), text.to_string(), None)
            .await
            .map_err(|e| EmbeddingError::Api(e.to_string()))?;

        // Convert f64 to f32
        let embedding: Vec<f32> = response.embeddings
            .into_iter()
            .flat_map(|v| v.into_iter().map(|x| x as f32))
            .collect();

        Ok(embedding)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
