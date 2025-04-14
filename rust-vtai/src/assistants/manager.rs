use std::sync::Arc;
use async_openai::{
    config::OpenAIConfig,
    Client,
    types::{
        AssistantObject,
        CreateAssistantRequest,
        CreateThreadRequest,
        CreateMessageRequest,
        CreateRunRequest,
        RunObject,
        ThreadObject,
        MessageObject,
        ToolsOutputs,
    },
};
use serde_json::Value;

use crate::utils::error::{Result, VTError};
use crate::utils::logger;
use crate::tools::registry::ToolRegistry;

/// Manager for OpenAI assistants
pub struct AssistantManager {
    /// OpenAI client
    client: Arc<Client<OpenAIConfig>>,

    /// Tool registry
    tool_registry: Arc<ToolRegistry>,

    /// Default assistant ID
    assistant_id: Option<String>,
}

impl AssistantManager {
    /// Create a new assistant manager
    pub fn new(client: Arc<Client<OpenAIConfig>>, tool_registry: Arc<ToolRegistry>, assistant_id: Option<String>) -> Self {
        Self {
            client,
            tool_registry,
            assistant_id,
        }
    }

    /// Get the default assistant ID
    pub fn get_assistant_id(&self) -> Result<String> {
        self.assistant_id.clone().ok_or_else(|| {
            VTError::ConfigError("No default assistant ID configured".to_string())
        })
    }

    /// Create a new thread
    pub async fn create_thread(&self) -> Result<ThreadObject> {
        logger::info("Creating new thread");

        let request = CreateThreadRequest::default();

        let thread = self.client.threads()
            .create(request)
            .await
            .map_err(|e| VTError::OpenAIError(format!("Failed to create thread: {}", e)))?;

        logger::info(&format!("Created thread: {}", thread.id));

        Ok(thread)
    }

    /// Add a message to a thread
    pub async fn add_message(&self, thread_id: &str, content: &str) -> Result<MessageObject> {
        logger::info(&format!("Adding message to thread {}: {}", thread_id, content));

        let request = CreateMessageRequest {
            role: "user".to_string(),
            content: content.to_string(),
            file_ids: None, // TODO: Support file attachments
            metadata: None,
        };

        let message = self.client.threads()
            .messages(thread_id)
            .create(request)
            .await
            .map_err(|e| VTError::OpenAIError(format!("Failed to add message: {}", e)))?;

        logger::info(&format!("Added message: {}", message.id));

        Ok(message)
    }

    /// Create a run with the default assistant
    pub async fn create_run(&self, thread_id: &str) -> Result<RunObject> {
        let assistant_id = self.get_assistant_id()?;

        logger::info(&format!("Creating run for thread {} with assistant {}", thread_id, assistant_id));

        let request = CreateRunRequest {
            assistant_id,
            instructions: None,
            tools: None,
            metadata: None,
            model: None,
        };

        let run = self.client.threads()
            .runs(thread_id)
            .create(request)
            .await
            .map_err(|e| VTError::OpenAIError(format!("Failed to create run: {}", e)))?;

        logger::info(&format!("Created run: {}", run.id));

        Ok(run)
    }

    /// Get a run's status
    pub async fn get_run(&self, thread_id: &str, run_id: &str) -> Result<RunObject> {
        logger::info(&format!("Getting run {} for thread {}", run_id, thread_id));

        let run = self.client.threads()
            .runs(thread_id)
            .retrieve(run_id)
            .await
            .map_err(|e| VTError::OpenAIError(format!("Failed to get run: {}", e)))?;

        logger::info(&format!("Run {} status: {:?}", run_id, run.status));

        Ok(run)
    }

    /// Get messages from a thread
    pub async fn get_messages(&self, thread_id: &str) -> Result<Vec<MessageObject>> {
        logger::info(&format!("Getting messages for thread {}", thread_id));

        let response = self.client.threads()
            .messages(thread_id)
            .list::<()>(&())
            .await
            .map_err(|e| VTError::OpenAIError(format!("Failed to get messages: {}", e)))?;

        logger::info(&format!("Got {} messages", response.data.len()));

        Ok(response.data)
    }

    /// Submit tool outputs for a run that requires action
    pub async fn submit_tool_outputs(&self, thread_id: &str, run_id: &str, tool_outputs: Vec<Value>) -> Result<RunObject> {
        logger::info(&format!("Submitting tool outputs for run {} in thread {}", run_id, thread_id));

        // Convert Vec<Value> to Vec<ToolsOutputs>
        let tool_outputs: Vec<ToolsOutputs> = tool_outputs.into_iter().filter_map(|v| serde_json::from_value(v).ok()).collect();

        let req = async_openai::types::SubmitToolOutputsRunRequest {
            tool_outputs,
        };
        let run = self.client.threads()
            .runs(thread_id)
            .submit_tool_outputs(run_id, req)
            .await
            .map_err(|e| VTError::OpenAIError(format!("Failed to submit tool outputs: {}", e)))?;

        logger::info(&format!("Submitted tool outputs, new status: {:?}", run.status));

        Ok(run)
    }
}