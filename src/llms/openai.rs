//! OpenAI LLM provider.

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionRequestAssistantMessageArgs,
        CreateChatCompletionRequestArgs, ResponseFormat
    },
    Client,
};
use async_trait::async_trait;

use super::traits::{GenerateOptions, LLM};
use crate::config::OpenAILLMConfig;
use crate::errors::LLMError;
use crate::models::{Message, Role};

/// OpenAI LLM provider
pub struct OpenAILLM {
    client: Client<OpenAIConfig>,
    model: String,
    default_temperature: f32,
    default_max_tokens: Option<u32>,
}

impl OpenAILLM {
    /// Create a new OpenAI LLM
    pub fn new(config: OpenAILLMConfig) -> Result<Self, LLMError> {
        let mut openai_config = OpenAIConfig::new();

        if let Some(api_key) = &config.api_key {
            openai_config = openai_config.with_api_key(api_key);
        }

        if let Some(base_url) = &config.base_url {
            openai_config = openai_config.with_api_base(base_url);
        }

        let client = Client::with_config(openai_config);

        Ok(Self {
            client,
            model: config.model,
            default_temperature: config.temperature,
            default_max_tokens: config.max_tokens,
        })
    }

    /// Convert Message to OpenAI message
    fn to_openai_message(msg: &Message) -> Result<ChatCompletionRequestMessage, LLMError> {
        match msg.role {
            Role::System => Ok(ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(msg.content.clone())
                    .build()
                    .map_err(|e| LLMError::Api(e.to_string()))?,
            )),
            Role::User => Ok(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(msg.content.clone())
                    .build()
                    .map_err(|e| LLMError::Api(e.to_string()))?,
            )),
            Role::Assistant => Ok(ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .content(msg.content.clone())
                    .build()
                    .map_err(|e| LLMError::Api(e.to_string()))?,
            )),
        }
    }
}

#[async_trait]
impl LLM for OpenAILLM {
    async fn generate(
        &self,
        messages: &[Message],
        options: GenerateOptions,
    ) -> Result<String, LLMError> {
        let openai_messages: Result<Vec<_>, _> = messages
            .iter()
            .map(Self::to_openai_message)
            .collect();
        let openai_messages = openai_messages?;

        let temperature = options.temperature.unwrap_or(self.default_temperature);
        let max_tokens = options.max_tokens.or(self.default_max_tokens);

        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder
            .model(&self.model)
            .messages(openai_messages)
            .temperature(temperature);

        if let Some(max) = max_tokens {
            request_builder.max_tokens(max);
        }

        if options.json_mode {
            request_builder.response_format(ResponseFormat::JsonObject);
        }

        let request = request_builder
            .build()
            .map_err(|e| LLMError::Api(e.to_string()))?;

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|e| LLMError::Api(e.to_string()))?;

        response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| LLMError::InvalidResponse("No content in response".to_string()))
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
