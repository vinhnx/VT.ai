use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;
use chrono::Utc;
use serde::{Serialize, Deserialize};
use async_openai::{
    types::{ChatCompletionRequestMessage, CreateChatCompletionRequest, Role},
    Client as OpenAIClient,
};

use crate::utils::error::{Result, VTError};
use crate::utils::logger;
use crate::utils::models::{Message, Settings};

/// Chat session storing messages and context
pub struct ChatSession {
    /// Unique session ID
    pub id: String,

    /// Messages in this session
    messages: Vec<Message>,

    /// Settings for this session
    pub settings: Settings,

    /// OpenAI thread ID if using assistant mode
    pub thread_id: Option<String>,

    /// Additional session variables
    pub variables: HashMap<String, String>,
}

impl ChatSession {
    /// Create a new chat session
    pub fn new(settings: Settings) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            messages: Vec::new(),
            settings,
            thread_id: None,
            variables: HashMap::new(),
        }
    }

    /// Add a message to the session
    pub fn add_message(&mut self, content: String, role: String) -> Message {
        let message = Message {
            id: Uuid::new_v4().to_string(),
            content,
            role,
            created_at: Utc::now().timestamp(),
            file_ids: None,
        };

        self.messages.push(message.clone());
        message
    }

    /// Get all messages in the session
    pub fn get_messages(&self) -> &[Message] {
        &self.messages
    }

    /// Update the session settings
    pub fn update_settings(&mut self, settings: Settings) {
        self.settings = settings;
    }
}

/// Chat Manager to handle multiple chat sessions
pub struct ChatManager {
    /// Map of session IDs to chat sessions
    sessions: Arc<Mutex<HashMap<String, ChatSession>>>,
}

impl ChatManager {
    /// Create a new chat manager
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new chat session
    pub async fn create_session(&self, settings: Settings) -> String {
        let session = ChatSession::new(settings);
        let session_id = session.id.clone();

        let mut sessions = self.sessions.lock().await;
        sessions.insert(session_id.clone(), session);

        session_id
    }

    /// Get a chat session by ID
    pub async fn get_session(&self, session_id: &str) -> Option<ChatSession> {
        let sessions = self.sessions.lock().await;
        sessions.get(session_id).cloned()
    }

    /// Process a user message and generate a response
    pub async fn process_message(
        &self,
        session_id: &str,
        content: String,
        client: &OpenAIClient,
    ) -> Result<Message> {
        // Get the session
        let mut sessions = self.sessions.lock().await;
        let session = sessions.get_mut(session_id).ok_or_else(|| {
            VTError::InvalidRequest(format!("Session not found: {}", session_id))
        })?;

        // Add the user message
        let user_message = session.add_message(content, "user".to_string());

        // Convert the messages to OpenAI format
        let mut openai_messages = Vec::new();

        // Add system message if not present
        if !session.messages.iter().any(|m| m.role == "system") {
            openai_messages.push(ChatCompletionRequestMessage {
                role: Role::System,
                content: "You are a helpful assistant.".into(),
                name: None,
                function_call: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Add all other messages
        for message in &session.messages {
            let role = match message.role.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "system" => Role::System,
                _ => continue, // Skip unknown roles
            };

            openai_messages.push(ChatCompletionRequestMessage {
                role,
                content: message.content.clone().into(),
                name: None,
                function_call: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Create the chat completion request
        let request = CreateChatCompletionRequest {
            model: map_model_name(&session.settings.chat_model),
            messages: openai_messages,
            temperature: Some(session.settings.temperature.into()),
            top_p: Some(session.settings.top_p.into()),
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            n: None,
            stream: None,
            stop: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            seed: None,
            function_call: None,
            functions: None,
        };

        // Call the OpenAI API
        logger::info(&format!("Sending request to model: {}", &session.settings.chat_model));

        match client.chat().create(request).await {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    if let Some(content) = &choice.message.content {
                        // Add the assistant's response to the session
                        let assistant_message = session.add_message(content.clone(), "assistant".to_string());
                        return Ok(assistant_message);
                    }
                }

                Err(VTError::InvalidRequest("No response from model".to_string()))
            },
            Err(e) => {
                logger::error(&format!("Error calling OpenAI API: {}", e));
                Err(VTError::from(e))
            }
        }
    }
}

/// Map friendly model names to actual API model names
fn map_model_name(model: &str) -> String {
    use crate::utils::constants::MODEL_ALIAS_MAP;

    MODEL_ALIAS_MAP.get(model).map_or_else(
        || model.to_string(),
        |&m| m.to_string()
    )
}