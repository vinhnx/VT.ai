use std::sync::Arc;
use crate::assistants::thread::{ThreadStore, ThreadHandler};
use crate::assistants::manager::AssistantManager;
use crate::tools::code::run_python_code;
use crate::tools::file::process_file;
use crate::tools::search::search_web;
use crate::tools::tavily::tavily_search;
use crate::utils::error::{Result, VTError};
use crate::utils::models::AppConfig;
use async_openai::types::{RunObject, ThreadObject};
use serde_json::Value;

pub struct AssistantRun<'a> {
    pub thread_store: &'a ThreadStore,
    pub manager: Arc<AssistantManager>,
    pub config: &'a AppConfig,
}

impl<'a> AssistantRun<'a> {
    pub fn new(thread_store: &'a ThreadStore, manager: Arc<AssistantManager>, config: &'a AppConfig) -> Self {
        Self { thread_store, manager, config }
    }

    pub async fn run(&self, thread_id: &str, user_query: &str) -> Result<String> {
        // Add user message to thread
        let handler = ThreadHandler::new(self.manager.clone(), ThreadObject { id: thread_id.to_string(), ..Default::default() });
        handler.add_message(user_query).await?;
        // Create a run and wait for completion (with tool call processing)
        let run = handler.run_and_wait(60).await?;
        // Get latest assistant message
        let messages = handler.get_latest_messages(1).await?;
        let response = messages.first().map(|m| m.content.clone()).unwrap_or_default();
        Ok(response)
    }

    // Tool call dispatcher (expand as needed)
    pub async fn process_tool_call(&self, tool_type: &str, args: &Value) -> Result<String> {
        match tool_type {
            "code_interpreter" => {
                let code = args.get("code").and_then(|v| v.as_str()).unwrap_or("");
                run_python_code(code).map_err(|e| VTError::Internal(e))
            }
            "file" => {
                let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
                process_file(path).map_err(|e| VTError::Internal(e))
            }
            "search" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                Ok(search_web(query).await.unwrap_or_default())
            }
            "tavily_search" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                tavily_search(query, self.config).await.map_err(|e| VTError::Internal(e))
            }
            _ => Err(VTError::InvalidRequest(format!("Unknown tool type: {}", tool_type)))
        }
    }
}
