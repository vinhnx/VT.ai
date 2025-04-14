/// Application-wide constants
use std::collections::HashMap;
use once_cell::sync::Lazy;

/// Application name
pub const APP_NAME: &str = "VT";

/// Default chat model to use
pub const DEFAULT_MODEL: &str = "gpt-4.1-mini";

/// Default temperature setting
pub const DEFAULT_TEMPERATURE: f32 = 0.8;

/// Default top-p setting
pub const DEFAULT_TOP_P: f32 = 1.0;

/// Default image generation model
pub const DEFAULT_IMAGE_GEN_MODEL: &str = "dall-e-3";

/// Default vision model
pub const DEFAULT_VISION_MODEL: &str = "gemini/gemini-2.0-flash";

/// Default TTS model
pub const DEFAULT_TTS_MODEL: &str = "gpt-4o-mini-tts";

/// Default TTS preset
pub const DEFAULT_TTS_PRESET: &str = "nova";

/// Default values for settings
pub const SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE: bool = true;
pub const SETTINGS_TRIMMED_MESSAGES_DEFAULT_VALUE: bool = true;
pub const SETTINGS_ENABLE_TTS_RESPONSE_DEFAULT_VALUE: bool = true;
pub const SETTINGS_USE_THINKING_MODEL_DEFAULT_VALUE: bool = false;

// Remove any hardcoded MODEL_ALIAS_MAP here and always use the one from models_map.rs
// To access model aliases, use crate::utils::models_map::MODEL_ALIAS_MAP

/// Setting key for chat model
pub const SETTINGS_CHAT_MODEL: &str = "settings_chat_model";

/// Setting key for temperature
pub const SETTINGS_TEMPERATURE: &str = "settings_temperature";

/// Setting key for top-p
pub const SETTINGS_TOP_P: &str = "settings_top_p";

/// Setting key for trimmed messages
pub const SETTINGS_TRIMMED_MESSAGES: &str = "settings_trimmed_messages";

/// Setting key for vision model
pub const SETTINGS_VISION_MODEL: &str = "settings_vision_model";

/// Setting key for image generation LLM model
pub const SETTINGS_IMAGE_GEN_LLM_MODEL: &str = "settings_image_gen_llm_model";

/// Setting key for image generation image style
pub const SETTINGS_IMAGE_GEN_IMAGE_STYLE: &str = "settings_image_gen_image_style";

/// Setting key for image generation image quality
pub const SETTINGS_IMAGE_GEN_IMAGE_QUALITY: &str = "settings_image_gen_image_quality";

/// Setting key for TTS model
pub const SETTINGS_TTS_MODEL: &str = "settings_tts_model";

/// Setting key for TTS voice preset model
pub const SETTINGS_TTS_VOICE_PRESET_MODEL: &str = "settings_tts_voice_preset_model";

/// Setting key for enabling TTS response
pub const SETTINGS_ENABLE_TTS_RESPONSE: &str = "settings_enable_tts_response";

/// Setting key for using dynamic conversation routing
pub const SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING: &str = "settings_use_dynamic_conversation_routing";

/// Setting key for using thinking model
pub const SETTINGS_USE_THINKING_MODEL: &str = "settings_use_thinking_model";

/// Error messages
pub const ERROR_API_UNAVAILABLE: &str = "API service is currently unavailable. Please try again later.";
pub const ERROR_MODEL_UNAVAILABLE: &str = "The selected model is currently unavailable. Please choose a different model.";
pub const ERROR_INVALID_REQUEST: &str = "Invalid request. Please check your input and try again.";
pub const ERROR_AUTHENTICATION: &str = "Authentication error. Please check your API keys.";
pub const ERROR_RATE_LIMIT: &str = "Rate limit exceeded. Please try again later.";
pub const ERROR_CONTEXT_LENGTH: &str = "Input is too long for this model. Please try a shorter message or a different model.";
pub const ERROR_CONTENT_FILTER: &str = "Content was filtered due to safety concerns. Please modify your request and try again.";
pub const ERROR_INTERNAL: &str = "An internal error occurred. Please try again later.";

/// File paths
pub const ROUTER_LAYERS_FILE: &str = "layers.json";

/// Resource paths
pub const RESOURCE_PATH: &str = "resources";
pub const LOGO_FILE: &str = "vt.jpg";