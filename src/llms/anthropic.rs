//! Anthropic Claude LLM provider.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::traits::{GenerateOptions, LLM};
use crate::config::AnthropicConfig;
use crate::errors::LLMError;
use crate::models::{Message, Role};

/// Anthropic Claude LLM provider
pub struct AnthropicLLM {
    client: Client,
    api_key: String,
    model: String,
    default_temperature: f32,
    default_max_tokens: u32,
}

impl AnthropicLLM {
    /// Create a new Anthropic LLM
    pub fn new(config: AnthropicConfig) -> Result<Self, LLMError> {
        let api_key = config
            .api_key
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .ok_or_else(|| LLMError::Api("ANTHROPIC_API_KEY not set".to_string()))?;

        let client = Client::new();

        Ok(Self {
            client,
            api_key,
            model: config.model,
            default_temperature: config.temperature,
            default_max_tokens: config.max_tokens,
        })
    }
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[async_trait]
impl LLM for AnthropicLLM {
    async fn generate(
        &self,
        messages: &[Message],
        options: GenerateOptions,
    ) -> Result<String, LLMError> {
        // Extract system message if present
        let mut system_message: Option<String> = None;
        let mut anthropic_messages = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    system_message = Some(msg.content.clone());
                }
                Role::User => {
                    anthropic_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: msg.content.clone(),
                    });
                }
                Role::Assistant => {
                    anthropic_messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: msg.content.clone(),
                    });
                }
            }
        }

        let temperature = options.temperature.unwrap_or(self.default_temperature);
        let max_tokens = options.max_tokens.unwrap_or(self.default_max_tokens);

        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens,
            system: system_message,
            messages: anthropic_messages,
            temperature: Some(temperature),
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| LLMError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError::Api(format!("Anthropic API error: {}", error_text)));
        }

        let result: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

        result
            .content
            .into_iter()
            .find(|c| c.content_type == "text")
            .and_then(|c| c.text)
            .ok_or_else(|| LLMError::InvalidResponse("No text content in response".to_string()))
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
