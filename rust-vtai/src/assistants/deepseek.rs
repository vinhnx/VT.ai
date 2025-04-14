use crate::utils::error::Result;

pub async fn chat_with_deepseek(message: &str) -> Result<String> {
    Ok(format!("[DeepSeek Placeholder] Response to: {}", message))
}