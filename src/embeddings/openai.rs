//! OpenAI embeddings provider.

use async_openai::{
    config::OpenAIConfig,
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
    Client,
};
use async_trait::async_trait;

use super::traits::Embedder;
use crate::config::OpenAIEmbedderConfig;
use crate::errors::EmbeddingError;

/// OpenAI embeddings provider
pub struct OpenAIEmbedder {
    client: Client<OpenAIConfig>,
    model: String,
    dimensions: usize,
}

impl OpenAIEmbedder {
    /// Create a new OpenAI embedder
    pub fn new(config: OpenAIEmbedderConfig) -> Result<Self, EmbeddingError> {
        let mut openai_config = OpenAIConfig::new();

        if let Some(api_key) = &config.api_key {
            openai_config = openai_config.with_api_key(api_key);
        }

        if let Some(base_url) = &config.base_url {
            openai_config = openai_config.with_api_base(base_url);
        }

        let client = Client::with_config(openai_config);

        let dimensions = config.dimensions.unwrap_or_else(|| {
            match config.model.as_str() {
                "text-embedding-3-large" => 3072,
                "text-embedding-3-small" => 1536,
                "text-embedding-ada-002" => 1536,
                _ => 1536,
            }
        });

        Ok(Self {
            client,
            model: config.model,
            dimensions,
        })
    }
}

#[async_trait]
impl Embedder for OpenAIEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(EmbeddingInput::String(text.to_string()))
            .build()
            .map_err(|e| EmbeddingError::Api(e.to_string()))?;

        let response = self
            .client
            .embeddings()
            .create(request)
            .await
            .map_err(|e| EmbeddingError::Api(e.to_string()))?;

        response
            .data
            .first()
            .map(|e| e.embedding.clone())
            .ok_or_else(|| EmbeddingError::InvalidResponse("No embedding in response".to_string()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let input: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(EmbeddingInput::StringArray(input))
            .build()
            .map_err(|e| EmbeddingError::Api(e.to_string()))?;

        let response = self
            .client
            .embeddings()
            .create(request)
            .await
            .map_err(|e| EmbeddingError::Api(e.to_string()))?;

        Ok(response.data.into_iter().map(|e| e.embedding).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
