//! Ollama LLM provider.

use async_trait::async_trait;
use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};

use super::traits::{GenerateOptions, LLM};
use crate::config::OllamaLLMConfig;
use crate::errors::LLMError;
use crate::models::{Message, Role};

/// Ollama LLM provider for local models
pub struct OllamaLLM {
    client: Ollama,
    model: String,
    default_temperature: f32,
}

impl OllamaLLM {
    /// Create a new Ollama LLM
    pub fn new(config: OllamaLLMConfig) -> Self {
        let url = url::Url::parse(&config.base_url).unwrap_or_else(|_| {
            url::Url::parse("http://localhost:11434").unwrap()
        });

        let host = url.host_str().unwrap_or("localhost").to_string();
        let port = url.port().unwrap_or(11434);

        let client = Ollama::new(host, port);

        Self {
            client,
            model: config.model,
            default_temperature: config.temperature,
        }
    }

    /// Format messages into a prompt string
    fn format_messages(messages: &[Message]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    prompt.push_str(&format!("System: {}\n\n", msg.content));
                }
                Role::User => {
                    prompt.push_str(&format!("User: {}\n\n", msg.content));
                }
                Role::Assistant => {
                    prompt.push_str(&format!("Assistant: {}\n\n", msg.content));
                }
            }
        }

        prompt.push_str("Assistant: ");
        prompt
    }
}

#[async_trait]
impl LLM for OllamaLLM {
    async fn generate(
        &self,
        messages: &[Message],
        options: GenerateOptions,
    ) -> Result<String, LLMError> {
        let prompt = Self::format_messages(messages);

        let mut request = GenerationRequest::new(self.model.clone(), prompt);

        // Set options
        let temperature = options.temperature.unwrap_or(self.default_temperature);
        request = request.options(
            ollama_rs::generation::options::GenerationOptions::default()
                .temperature(temperature as f64),
        );

        if options.json_mode {
            request = request.format(ollama_rs::generation::completion::request::FormatType::Json);
        }

        let response = self
            .client
            .generate(request)
            .await
            .map_err(|e| LLMError::Api(e.to_string()))?;

        Ok(response.response)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
