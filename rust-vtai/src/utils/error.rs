use std::fmt;
use thiserror::Error;

/// Custom result type with our error
pub type Result<T> = std::result::Result<T, VTError>;

/// Error types for VT.ai
#[derive(Error, Debug)]
pub enum VTError {
    /// Authentication errors
    #[error("Authentication error: {0}")]
    Authentication(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// File errors
    #[error("File error: {0}")]
    FileError(String),

    /// Invalid request errors
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Network errors
    #[error("Network error: {0}")]
    Network(String),

    /// OpenAI API errors
    #[error("OpenAI API error: {0}")]
    OpenAIError(String),

    /// Rate limit errors
    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    /// Content filter errors
    #[error("Content filtered: {0}")]
    ContentFiltered(String),

    /// Internal errors
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Map IO errors to VTError
impl From<std::io::Error> for VTError {
    fn from(error: std::io::Error) -> Self {
        VTError::FileError(error.to_string())
    }
}

/// Map JSON errors to VTError
impl From<serde_json::Error> for VTError {
    fn from(error: serde_json::Error) -> Self {
        VTError::InvalidRequest(error.to_string())
    }
}

/// Map OpenAI errors to VTError
impl From<async_openai::error::OpenAIError> for VTError {
    fn from(error: async_openai::error::OpenAIError) -> Self {
        match error {
            async_openai::error::OpenAIError::ApiError(e) => {
                match e.code.as_ref().and_then(|v| v.as_str()) {
                    Some("authentication_error") => VTError::Authentication(e.message),
                    Some("rate_limit_exceeded") => VTError::RateLimit(e.message),
                    Some("content_filter") => VTError::ContentFiltered(e.message),
                    _ => VTError::OpenAIError(e.message),
                }
            },
            async_openai::error::OpenAIError::StreamError(e) => VTError::Network(e.to_string()),
            _ => VTError::Internal(error.to_string()),
        }
    }
}

/// Map reqwest errors to VTError
impl From<reqwest::Error> for VTError {
    fn from(error: reqwest::Error) -> Self {
        VTError::Network(error.to_string())
    }
}