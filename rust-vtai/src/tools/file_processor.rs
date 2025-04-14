use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use tokio::fs;
use std::path::PathBuf;
use uuid::Uuid;

use crate::tools::registry::Tool;
use crate::utils::error::{Result, VTError};
use crate::utils::logger;

/// File processor tool for handling file operations
pub struct FileProcessor {
    /// Base directory for file operations
    base_dir: PathBuf,
}

/// Input parameters for reading a file
#[derive(Debug, Deserialize)]
pub struct ReadFileParams {
    /// The path to the file
    pub path: String,
}

/// Input parameters for writing a file
#[derive(Debug, Deserialize)]
pub struct WriteFileParams {
    /// The path to the file
    pub path: String,

    /// The content to write to the file
    pub content: String,
}

/// Input parameters for listing directory contents
#[derive(Debug, Deserialize)]
pub struct ListDirParams {
    /// The path to the directory
    pub path: String,
}

impl FileProcessor {
    /// Create a new file processor
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            base_dir,
        }
    }

    /// Resolve a path relative to the base directory
    fn resolve_path(&self, path: &str) -> Result<PathBuf> {
        let path = PathBuf::from(path);

        // Make sure the path doesn't escape the base directory
        let resolved = self.base_dir.join(path);
        let canonicalized = resolved.canonicalize().map_err(|e| {
            VTError::FileError(format!("Failed to resolve path: {}", e))
        })?;

        if !canonicalized.starts_with(&self.base_dir) {
            return Err(VTError::InvalidRequest("Path escapes the base directory".to_string()));
        }

        Ok(canonicalized)
    }

    /// Read a file
    async fn read_file(&self, params: ReadFileParams) -> Result<String> {
        let path = self.resolve_path(&params.path)?;

        // Check if the file exists
        if !path.exists() {
            return Err(VTError::FileError(format!("File not found: {}", params.path)));
        }

        // Read the file
        let content = fs::read_to_string(&path).await.map_err(|e| {
            VTError::FileError(format!("Failed to read file: {}", e))
        })?;

        Ok(content)
    }

    /// Write a file
    async fn write_file(&self, params: WriteFileParams) -> Result<String> {
        let path = self.resolve_path(&params.path)?;

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                VTError::FileError(format!("Failed to create directories: {}", e))
            })?;
        }

        // Write the file
        fs::write(&path, &params.content).await.map_err(|e| {
            VTError::FileError(format!("Failed to write file: {}", e))
        })?;

        Ok(format!("Successfully wrote {} bytes to {}", params.content.len(), params.path))
    }

    /// List directory contents
    async fn list_dir(&self, params: ListDirParams) -> Result<String> {
        let path = self.resolve_path(&params.path)?;

        // Check if the directory exists
        if !path.exists() {
            return Err(VTError::FileError(format!("Directory not found: {}", params.path)));
        }

        // Check if it's a directory
        if !path.is_dir() {
            return Err(VTError::FileError(format!("Not a directory: {}", params.path)));
        }

        // List the directory contents
        let mut entries = fs::read_dir(&path).await.map_err(|e| {
            VTError::FileError(format!("Failed to read directory: {}", e))
        })?;

        let mut result = Vec::new();

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            VTError::FileError(format!("Failed to read directory entry: {}", e))
        })? {
            let file_type = entry.file_type().await.map_err(|e| {
                VTError::FileError(format!("Failed to get file type: {}", e))
            })?;

            let name = entry.file_name().to_string_lossy().to_string();

            if file_type.is_dir() {
                result.push(format!("{}/", name));
            } else {
                result.push(name);
            }
        }

        // Sort the results
        result.sort();

        Ok(result.join("\n"))
    }
}

#[async_trait]
impl Tool for FileProcessor {
    fn name(&self) -> &str {
        "file_processor"
    }

    fn description(&self) -> &str {
        "Performs file operations like reading, writing, and listing directory contents"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "The operation to perform (read_file, write_file, list_dir)",
                    "enum": ["read_file", "write_file", "list_dir"]
                },
                "path": {
                    "type": "string",
                    "description": "The path to the file or directory"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file (for write_file operation)"
                }
            },
            "required": ["operation", "path"],
            "allOf": [
                {
                    "if": {
                        "properties": { "operation": { "const": "write_file" } }
                    },
                    "then": {
                        "required": ["content"]
                    }
                }
            ]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        // Get the operation
        let operation = args.get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| VTError::InvalidRequest("Missing operation parameter".to_string()))?;

        // Execute the appropriate operation
        match operation {
            "read_file" => {
                let params: ReadFileParams = serde_json::from_value(args)
                    .map_err(|e| VTError::InvalidRequest(format!("Invalid arguments: {}", e)))?;
                self.read_file(params).await
            },
            "write_file" => {
                let params: WriteFileParams = serde_json::from_value(args)
                    .map_err(|e| VTError::InvalidRequest(format!("Invalid arguments: {}", e)))?;
                self.write_file(params).await
            },
            "list_dir" => {
                let params: ListDirParams = serde_json::from_value(args)
                    .map_err(|e| VTError::InvalidRequest(format!("Invalid arguments: {}", e)))?;
                self.list_dir(params).await
            },
            _ => Err(VTError::InvalidRequest(format!("Unsupported operation: {}", operation))),
        }
    }
}