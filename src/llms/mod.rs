//! LLM providers for mem0-rust.
//!
//! This module provides various LLM backends for fact extraction:
//! - OpenAI (GPT-4o, GPT-4o-mini)
//! - Ollama (local models)
//! - Anthropic (Claude)

mod traits;

pub use traits::{generate_json, GenerateOptions, LLM};

#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openai")]
pub use openai::OpenAILLM;

#[cfg(feature = "ollama")]
mod ollama;
#[cfg(feature = "ollama")]
pub use ollama::OllamaLLM;

#[cfg(feature = "anthropic")]
mod anthropic;
#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicLLM;

use crate::config::LLMConfig;
use crate::errors::LLMError;
use std::sync::Arc;

/// Create an LLM from configuration
#[allow(unused_variables)]
pub fn create_llm(config: &LLMConfig) -> Result<Arc<dyn LLM>, LLMError> {
    #[cfg(feature = "openai")]
    if let LLMConfig::OpenAI(cfg) = config {
        return Ok(Arc::new(OpenAILLM::new(cfg.clone())?));
    }

    #[cfg(feature = "ollama")]
    if let LLMConfig::Ollama(cfg) = config {
        return Ok(Arc::new(OllamaLLM::new(cfg.clone())));
    }

    #[cfg(feature = "anthropic")]
    if let LLMConfig::Anthropic(cfg) = config {
        return Ok(Arc::new(AnthropicLLM::new(cfg.clone())?));
    }

    Err(LLMError::NotConfigured)
}
