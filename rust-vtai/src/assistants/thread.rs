use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use async_openai::types::{MessageObject, RunObject, RunStatus, ThreadObject};
use serde_json::Value;
use std::collections::HashMap;
use tokio::sync::Mutex;

use crate::assistants::manager::AssistantManager;
use crate::utils::error::{Result, VTError};
use crate::utils::logger;

#[derive(Debug, Clone)]
pub struct Thread {
    pub id: String,
    pub messages: Vec<String>,
}

#[derive(Default)]
pub struct ThreadStore {
    pub threads: Mutex<HashMap<String, Thread>>,
}

impl ThreadStore {
    pub async fn create_thread(&self, id: String) {
        let mut threads = self.threads.lock().await;
        threads.insert(id.clone(), Thread { id, messages: vec![] });
    }
    pub async fn add_message(&self, thread_id: &str, message: String) {
        let mut threads = self.threads.lock().await;
        if let Some(thread) = threads.get_mut(thread_id) {
            thread.messages.push(message);
        }
    }
    pub async fn get_thread(&self, thread_id: &str) -> Option<Thread> {
        let threads = self.threads.lock().await;
        threads.get(thread_id).cloned()
    }
}

/// Handler for thread message processing
pub struct ThreadHandler {
    /// The assistant manager
    manager: Arc<AssistantManager>,

    /// The thread object
    thread: ThreadObject,
}

impl ThreadHandler {
    /// Create a new thread handler
    pub fn new(manager: Arc<AssistantManager>, thread: ThreadObject) -> Self {
        Self {
            manager,
            thread,
        }
    }

    /// Get the thread ID
    pub fn thread_id(&self) -> &str {
        &self.thread.id
    }

    /// Add a message to the thread
    pub async fn add_message(&self, content: &str) -> Result<MessageObject> {
        self.manager.add_message(self.thread_id(), content).await
    }

    /// Create a run and wait for it to complete
    pub async fn run_and_wait(&self, timeout_secs: u64) -> Result<RunObject> {
        // Create the run
        let run = self.manager.create_run(self.thread_id()).await?;

        // Wait for the run to complete
        self.wait_for_run(&run.id, timeout_secs).await
    }

    /// Wait for a run to complete
    pub async fn wait_for_run(&self, run_id: &str, timeout_secs: u64) -> Result<RunObject> {
        let start_time = std::time::Instant::now();
        let timeout = Duration::from_secs(timeout_secs);

        loop {
            // Check if we've exceeded the timeout
            if start_time.elapsed() > timeout {
                return Err(VTError::Internal(format!("Run timed out after {} seconds", timeout_secs)));
            }

            // Get the run status
            let run = self.manager.get_run(self.thread_id(), run_id).await?;

            match run.status {
                RunStatus::Completed => {
                    return Ok(run);
                },
                RunStatus::Failed => {
                    return Err(VTError::OpenAIError(format!("Run failed: {:?}", run.last_error)));
                },
                RunStatus::Cancelled => {
                    return Err(VTError::OpenAIError("Run was cancelled".to_string()));
                },
                RunStatus::Expired => {
                    return Err(VTError::OpenAIError("Run expired".to_string()));
                },
                RunStatus::RequiresAction => {
                    // Handle tool calls
                    if let Some(required_action) = &run.required_action {
                        if required_action.r#type == "submit_tool_outputs" {
                            let submit_tool = &required_action.submit_tool_outputs;
                            // Convert Vec<RunToolCallObject> to Vec<Value>
                            let tool_calls_json: Vec<Value> = submit_tool.tool_calls.iter().filter_map(|tc| serde_json::to_value(tc).ok()).collect();
                            let tool_outputs = self.process_tool_calls(&tool_calls_json).await?;

                            // Submit the tool outputs
                            self.manager.submit_tool_outputs(self.thread_id(), run_id, tool_outputs).await?;
                        }
                    }
                },
                _ => {
                    // Still in progress, wait a bit
                    sleep(Duration::from_millis(500)).await;
                }
            }
        }
    }

    /// Process tool calls
    async fn process_tool_calls(&self, tool_calls: &[Value]) -> Result<Vec<Value>> {
        // For now, just return empty tool outputs
        // In a real implementation, this would execute the actual tools

        let mut outputs = Vec::new();

        for tool_call in tool_calls {
            if let Some(id) = tool_call.get("id").and_then(|v| v.as_str()) {
                outputs.push(serde_json::json!({
                    "tool_call_id": id,
                    "output": "Mock tool output for now",
                }));
            }
        }

        Ok(outputs)
    }

    /// Get latest messages from the thread
    pub async fn get_latest_messages(&self, limit: usize) -> Result<Vec<MessageObject>> {
        let messages = self.manager.get_messages(self.thread_id()).await?;

        // Return the latest messages, up to the limit
        Ok(messages.into_iter().take(limit).collect())
    }
}