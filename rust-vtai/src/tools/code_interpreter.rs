use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::process::Command;
use std::sync::Arc;
use tempfile::NamedTempFile;
use std::io::Write;
use tokio::fs;
use uuid::Uuid;

use crate::tools::registry::Tool;
use crate::utils::error::{Result, VTError};
use crate::utils::logger;

/// Code interpreter tool for executing code snippets
pub struct CodeInterpreter {
    /// Temporary directory for code execution
    temp_dir: Arc<tempfile::TempDir>,
}

/// Input parameters for the code interpreter
#[derive(Debug, Deserialize)]
pub struct CodeInterpreterParams {
    /// The language to execute (python, javascript, etc.)
    pub language: String,

    /// The code to execute
    pub code: String,
}

impl CodeInterpreter {
    /// Create a new code interpreter
    pub fn new() -> Result<Self> {
        // Create a temporary directory for code execution
        let temp_dir = tempfile::tempdir()
            .map_err(|e| VTError::Internal(format!("Failed to create temp directory: {}", e)))?;

        logger::info(&format!("Created code interpreter temp directory: {}", temp_dir.path().display()));

        Ok(Self {
            temp_dir: Arc::new(temp_dir),
        })
    }

    /// Execute Python code
    async fn execute_python(&self, code: &str) -> Result<String> {
        // Create a temporary file for the Python code
        let file_path = self.temp_dir.path().join(format!("{}.py", Uuid::new_v4()));
        fs::write(&file_path, code).await?;

        // Execute the Python code
        let output = Command::new("python")
            .arg(&file_path)
            .output()
            .map_err(|e| VTError::Internal(format!("Failed to execute Python code: {}", e)))?;

        // Process the output
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !stderr.is_empty() {
            logger::warn(&format!("Python execution stderr: {}", stderr));
        }

        // Clean up
        fs::remove_file(file_path).await.ok();

        if !output.status.success() {
            return Err(VTError::InvalidRequest(format!("Python execution failed: {}", stderr)));
        }

        Ok(format!("Output:\n{}", stdout))
    }

    /// Execute JavaScript code
    async fn execute_javascript(&self, code: &str) -> Result<String> {
        // Create a temporary file for the JavaScript code
        let file_path = self.temp_dir.path().join(format!("{}.js", Uuid::new_v4()));
        fs::write(&file_path, code).await?;

        // Execute the JavaScript code with Node.js
        let output = Command::new("node")
            .arg(&file_path)
            .output()
            .map_err(|e| VTError::Internal(format!("Failed to execute JavaScript code: {}", e)))?;

        // Process the output
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !stderr.is_empty() {
            logger::warn(&format!("JavaScript execution stderr: {}", stderr));
        }

        // Clean up
        fs::remove_file(file_path).await.ok();

        if !output.status.success() {
            return Err(VTError::InvalidRequest(format!("JavaScript execution failed: {}", stderr)));
        }

        Ok(format!("Output:\n{}", stdout))
    }
}

#[async_trait]
impl Tool for CodeInterpreter {
    fn name(&self) -> &str {
        "code_interpreter"
    }

    fn description(&self) -> &str {
        "Executes code snippets in various programming languages"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "The programming language to use (python, javascript)",
                    "enum": ["python", "javascript"]
                },
                "code": {
                    "type": "string",
                    "description": "The code to execute"
                }
            },
            "required": ["language", "code"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        // Parse the arguments
        let params: CodeInterpreterParams = serde_json::from_value(args)
            .map_err(|e| VTError::InvalidRequest(format!("Invalid arguments: {}", e)))?;

        // Execute the code based on the language
        match params.language.as_str() {
            "python" => self.execute_python(&params.code).await,
            "javascript" => self.execute_javascript(&params.code).await,
            _ => Err(VTError::InvalidRequest(format!("Unsupported language: {}", params.language))),
        }
    }
}