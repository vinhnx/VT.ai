use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::utils::error::{Result, VTError};

/// Tool call with arguments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call
    pub id: String,

    /// Name of the tool to call
    pub name: String,

    /// Arguments to pass to the tool
    pub arguments: Value,
}

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// ID of the original tool call
    pub tool_call_id: String,

    /// Result of the tool execution
    pub output: String,

    /// Whether the tool execution was successful
    pub success: bool,

    /// Additional content or data (optional)
    pub content: Option<Value>,
}

/// Trait for tool implementations
#[async_trait]
pub trait Tool: Send + Sync {
    /// Name of the tool
    fn name(&self) -> &str;

    /// Description of what the tool does
    fn description(&self) -> &str;

    /// JSON schema describing the tool's parameters
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with the given arguments
    async fn execute(&self, args: Value) -> Result<String>;
}

/// Registry of available tools
pub struct ToolRegistry {
    /// Map of tool names to tool implementations
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new tool registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::new(tool));
    }

    /// Get a tool by name
    pub fn get_tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Get all registered tools
    pub fn all_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.values().cloned().collect()
    }

    /// Execute a tool call
    pub async fn execute_tool_call(&self, call: ToolCall) -> Result<ToolResult> {
        let tool = self.get_tool(&call.name).ok_or_else(|| {
            VTError::InvalidRequest(format!("Tool not found: {}", call.name))
        })?;

        match tool.execute(call.arguments).await {
            Ok(output) => {
                Ok(ToolResult {
                    tool_call_id: call.id,
                    output,
                    success: true,
                    content: None,
                })
            },
            Err(e) => {
                Ok(ToolResult {
                    tool_call_id: call.id,
                    output: format!("Error: {}", e),
                    success: false,
                    content: None,
                })
            }
        }
    }

    /// Generate OpenAI-compatible tool specifications
    pub fn get_tool_specs(&self) -> Vec<Value> {
        self.all_tools().iter().map(|tool| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.parameters_schema(),
                }
            })
        }).collect()
    }
}