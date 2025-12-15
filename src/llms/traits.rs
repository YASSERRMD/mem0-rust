//! LLM trait definition.

use async_trait::async_trait;

use crate::errors::LLMError;
use crate::models::Message;

/// Options for LLM generation
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    /// Temperature (0.0 to 1.0)
    pub temperature: Option<f32>,

    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,

    /// Force JSON output
    pub json_mode: bool,
}

/// Trait for LLM providers
#[async_trait]
pub trait LLM: Send + Sync {
    /// Generate a text response
    async fn generate(
        &self,
        messages: &[Message],
        options: GenerateOptions,
    ) -> Result<String, LLMError>;

    /// Get the model name
    fn model_name(&self) -> &str;
}

/// Generate and parse a JSON response (standalone function)
pub async fn generate_json<T: serde::de::DeserializeOwned>(
    llm: &dyn LLM,
    messages: &[Message],
    options: GenerateOptions,
) -> Result<T, LLMError> {
    let mut opts = options;
    opts.json_mode = true;

    let response = llm.generate(messages, opts).await?;

    // Try to extract JSON from response (handle markdown code blocks)
    let json_str = extract_json(&response);

    serde_json::from_str(&json_str)
        .map_err(|e| LLMError::JsonParse(format!("{}: {}", e, json_str)))
}

/// Extract JSON from response (handles markdown code blocks)
fn extract_json(response: &str) -> String {
    let response = response.trim();

    // Try to extract from ```json ... ``` blocks
    if let Some(start) = response.find("```json") {
        if let Some(end) = response[start + 7..].find("```") {
            return response[start + 7..start + 7 + end].trim().to_string();
        }
    }

    // Try to extract from ``` ... ``` blocks
    if let Some(start) = response.find("```") {
        if let Some(end) = response[start + 3..].find("```") {
            let content = response[start + 3..start + 3 + end].trim();
            // Skip language identifier if present
            if let Some(newline) = content.find('\n') {
                let first_line = &content[..newline];
                if !first_line.starts_with('{') && !first_line.starts_with('[') {
                    return content[newline + 1..].trim().to_string();
                }
            }
            return content.to_string();
        }
    }

    // Try to find raw JSON object or array
    if let Some(start) = response.find('{') {
        if let Some(end) = response.rfind('}') {
            return response[start..=end].to_string();
        }
    }

    if let Some(start) = response.find('[') {
        if let Some(end) = response.rfind(']') {
            return response[start..=end].to_string();
        }
    }

    response.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_raw() {
        let input = r#"{"key": "value"}"#;
        assert_eq!(extract_json(input), r#"{"key": "value"}"#);
    }

    #[test]
    fn test_extract_json_code_block() {
        let input = r#"```json
{"key": "value"}
```"#;
        assert_eq!(extract_json(input), r#"{"key": "value"}"#);
    }

    #[test]
    fn test_extract_json_with_text() {
        let input = r#"Here is the result: {"key": "value"} as requested."#;
        assert_eq!(extract_json(input), r#"{"key": "value"}"#);
    }
}
