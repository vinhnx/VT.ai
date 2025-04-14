use std::error::Error;
use crate::utils::models::AppConfig;
use async_openai::{Client, config::OpenAIConfig};
use async_openai::types::{CreateChatCompletionRequestArgs, ChatCompletionRequestMessage, Role, ChatCompletionRequestUserMessageArgs};

fn build_openai_client(config: &AppConfig) -> Client<OpenAIConfig> {
    let mut builder = OpenAIConfig::new();
    if let Some(ref api_key) = config.openai_api_key {
        builder = builder.with_api_key(api_key);
    }
    if let Some(ref base_url) = config.openai_api_base {
        builder = builder.with_api_base(base_url);
    }
    Client::with_config(builder)
}

pub async fn chat_with_openai(prompt: &str) -> Result<String, String> {
    // Placeholder: In production, call the OpenAI API
    Ok(format!("[OpenAI] Response to: {}", prompt))
}

pub async fn chat_with_openai_with_config(prompt: &str, config: &AppConfig) -> Result<String, String> {
    let client = build_openai_client(config);

    let user_message = ChatCompletionRequestUserMessageArgs::default()
        .content(prompt)
        .build()
        .map_err(|e| format!("Failed to build message: {}", e))?;

    let messages = vec![
        ChatCompletionRequestMessage::User(user_message)
    ];

    let request = match CreateChatCompletionRequestArgs::default()
        .model(config.default_model.clone())
        .messages(messages)
        .temperature(config.temperature)
        .top_p(config.top_p)
        .build() {
            Ok(req) => req,
            Err(e) => return Err(format!("Failed to build request: {}", e)),
        };

    match client.chat().create(request).await {
        Ok(response) => {
            if let Some(choice) = response.choices.first() {
                if let Some(content) = &choice.message.content {
                    Ok(content.clone())
                } else {
                    Err("No content in response choice".to_string())
                }
            } else {
                Err("No choices in response".to_string())
            }
        }
        Err(e) => Err(format!("API call failed: {}", e)),
    }
}