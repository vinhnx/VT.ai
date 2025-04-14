use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;

use crate::utils::constants::{
    DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
    DEFAULT_IMAGE_GEN_MODEL, DEFAULT_VISION_MODEL, DEFAULT_TTS_MODEL, DEFAULT_TTS_PRESET,
    SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE,
    SETTINGS_TRIMMED_MESSAGES_DEFAULT_VALUE,
    SETTINGS_ENABLE_TTS_RESPONSE_DEFAULT_VALUE,
    SETTINGS_USE_THINKING_MODEL_DEFAULT_VALUE,
};

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// OpenAI API key
    pub openai_api_key: Option<String>,

    /// Anthropic API key
    pub anthropic_api_key: Option<String>,

    /// Default chat model
    pub default_model: String,

    /// Default temperature
    pub temperature: f32,

    /// Default top_p
    pub top_p: f32,

    /// Default vision model
    pub default_vision_model: String,

    /// Default image generation model
    pub default_image_gen_model: String,

    /// Default text-to-speech model
    pub default_tts_model: String,

    /// Default text-to-speech preset
    pub default_tts_preset: String,

    /// Default dynamic routing
    pub default_dynamic_routing: bool,

    /// Default thinking model
    pub default_thinking_model: bool,

    /// Server port
    pub port: u16,

    /// Default assistant ID (if using OpenAI Assistants API)
    pub assistant_id: Option<String>,

    /// Enable file processing
    pub enable_file_processing: bool,

    /// Enable code interpreter
    pub enable_code_interpreter: bool,

    /// Enable web search
    pub enable_web_search: bool,

    /// OpenAI/LiteLLM API base URL
    pub openai_api_base: Option<String>,

    /// Tavily API key (for Tavily Search)
    pub tavily_api_key: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            openai_api_key: None,
            anthropic_api_key: None,
            default_model: DEFAULT_MODEL.to_string(),
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            default_vision_model: DEFAULT_VISION_MODEL.to_string(),
            default_image_gen_model: DEFAULT_IMAGE_GEN_MODEL.to_string(),
            default_tts_model: DEFAULT_TTS_MODEL.to_string(),
            default_tts_preset: DEFAULT_TTS_PRESET.to_string(),
            default_dynamic_routing: SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE,
            default_thinking_model: SETTINGS_USE_THINKING_MODEL_DEFAULT_VALUE,
            port: 8000,
            assistant_id: None,
            enable_file_processing: true,
            enable_code_interpreter: true,
            enable_web_search: true,
            openai_api_base: None,
            tavily_api_key: None,
        }
    }
}

impl AppConfig {
    pub fn from_env() -> Self {
        Self {
            openai_api_key: env::var("OPENAI_API_KEY").ok(),
            anthropic_api_key: env::var("ANTHROPIC_API_KEY").ok(),
            default_model: env::var("VT_DEFAULT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            temperature: env::var("TEMPERATURE").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_TEMPERATURE),
            top_p: env::var("TOP_P").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_TOP_P),
            default_vision_model: env::var("DEFAULT_VISION_MODEL").unwrap_or_else(|_| DEFAULT_VISION_MODEL.to_string()),
            default_image_gen_model: env::var("DEFAULT_IMAGE_GEN_MODEL").unwrap_or_else(|_| DEFAULT_IMAGE_GEN_MODEL.to_string()),
            default_tts_model: env::var("DEFAULT_TTS_MODEL").unwrap_or_else(|_| DEFAULT_TTS_MODEL.to_string()),
            default_tts_preset: env::var("DEFAULT_TTS_PRESET").unwrap_or_else(|_| DEFAULT_TTS_PRESET.to_string()),
            default_dynamic_routing: env::var("DEFAULT_DYNAMIC_ROUTING").map(|v| v == "1").unwrap_or(SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE),
            default_thinking_model: env::var("DEFAULT_THINKING_MODEL").map(|v| v == "1").unwrap_or(SETTINGS_USE_THINKING_MODEL_DEFAULT_VALUE),
            port: env::var("PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8000),
            assistant_id: env::var("ASSISTANT_ID").ok(),
            enable_file_processing: env::var("ENABLE_FILE_PROCESSING").map(|v| v == "1").unwrap_or(true),
            enable_code_interpreter: env::var("ENABLE_CODE_INTERPRETER").map(|v| v == "1").unwrap_or(true),
            enable_web_search: env::var("ENABLE_WEB_SEARCH").map(|v| v == "1").unwrap_or(true),
            openai_api_base: env::var("OPENAI_API_BASE").ok()
                .or_else(|| env::var("LITELLM_API_BASE").ok()),
            tavily_api_key: env::var("TAVILY_API_KEY").ok(),
        }
    }
}

/// User settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Chat model to use
    pub chat_model: String,

    /// Temperature setting
    pub temperature: f32,

    /// Top-p setting
    pub top_p: f32,

    /// Vision model
    pub vision_model: String,

    /// Image generation LLM model
    pub image_gen_llm_model: String,

    /// Image generation style
    pub image_gen_image_style: String,

    /// Image generation quality
    pub image_gen_image_quality: String,

    /// Text-to-speech model
    pub tts_model: String,

    /// Text-to-speech voice preset model
    pub tts_voice_preset_model: String,

    /// Enable text-to-speech response
    pub enable_tts_response: bool,

    /// Use dynamic conversation routing
    pub use_dynamic_conversation_routing: bool,

    /// Use thinking model
    pub use_thinking_model: bool,

    /// Trimmed messages
    pub trimmed_messages: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            chat_model: DEFAULT_MODEL.to_string(),
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            vision_model: DEFAULT_VISION_MODEL.to_string(),
            image_gen_llm_model: DEFAULT_IMAGE_GEN_MODEL.to_string(),
            image_gen_image_style: "vivid".to_string(),
            image_gen_image_quality: "standard".to_string(),
            tts_model: DEFAULT_TTS_MODEL.to_string(),
            tts_voice_preset_model: DEFAULT_TTS_PRESET.to_string(),
            enable_tts_response: SETTINGS_ENABLE_TTS_RESPONSE_DEFAULT_VALUE,
            use_dynamic_conversation_routing: SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE,
            use_thinking_model: SETTINGS_USE_THINKING_MODEL_DEFAULT_VALUE,
            trimmed_messages: SETTINGS_TRIMMED_MESSAGES_DEFAULT_VALUE,
        }
    }
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique message ID
    pub id: String,

    /// Message content
    pub content: String,

    /// Message role (user, assistant, system)
    pub role: String,

    /// Timestamp when the message was created
    pub created_at: i64,

    /// IDs of attached files (if any)
    pub file_ids: Option<Vec<String>>,
}

/// Chat profile for the application
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatProfile {
    /// The ID of the profile
    pub id: String,

    /// The display name of the profile
    pub name: String,

    /// The description of the profile
    pub description: String,

    /// The icon to use for the profile
    pub icon: Option<String>,

    /// Whether this is the default profile
    pub is_default: bool,

    /// Custom settings for this profile
    pub settings: HashMap<String, serde_json::Value>,
}

/// User session data
#[derive(Clone, Debug, Default)]
pub struct UserSession {
    /// The current settings
    pub settings: Settings,

    /// The current chat profile
    pub profile: Option<String>,

    /// The thread ID for assistant conversations
    pub thread_id: Option<String>,

    /// Map of session variables
    pub variables: HashMap<String, String>,
}

/// File metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    /// Unique file ID
    pub id: String,

    /// Original file name
    pub name: String,

    /// File type/MIME type
    pub file_type: String,

    /// File size in bytes
    pub size: usize,

    /// File path on disk
    #[serde(skip)]
    pub path: Option<String>,
}

/// Tool call information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    /// The ID of the tool call
    pub id: String,

    /// The type of the tool
    pub tool_type: String,

    /// The name of the function to call
    pub function_name: String,

    /// The arguments to pass to the function
    pub arguments: serde_json::Value,
}

/// Tool output
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolOutput {
    /// The ID of the tool call this output is for
    pub tool_call_id: String,

    /// The output of the tool call
    pub output: String,
}
