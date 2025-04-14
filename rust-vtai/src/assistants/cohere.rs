use std::result::Result;

pub async fn chat_with_cohere(message: &str) -> Result<String, String> {
    Ok(format!("[Cohere Placeholder] Response to: {}", message))
}