"""
Core application constants for VT.

This module contains constants used throughout the application,
including names, identifiers, and feature flags.
"""

from typing import Dict, Final

# Application names and identifiers
APP_NAME: Final[str] = "VT"
MINO_ASSISTANT_NAME: Final[str] = "Mino"

# Assistant tools
ASSISTANT_TOOL_CODE_INTERPRETER: Final[str] = "code_interpreter"
ASSISTANT_TOOL_RETRIEVAL: Final[str] = "retrieval"
ASSISTANT_TOOL_FUNCTION: Final[str] = "function"

# Feature flags
DEBUG_MODE: Final[bool] = False

# Model mappings
MODEL_ALIAS_MAP: Dict[str, str] = {
    # OpenAI models
    "OpenAI - GPT-o1": "o1",
    "OpenAI - GPT-o1 Mini": "o1-mini",
    "OpenAI - GPT-o1 Pro": "o1-pro",
    "OpenAI - GPT-o3 Mini": "o3-mini",
    "OpenAI - GPT-4.5 Preview": "gpt-4.5-preview",
    "OpenAI - GPT-4o": "gpt-4o",
    "OpenAI - GPT-4o Mini": "gpt-4o-mini",
    # Anthropic models
    "Anthropic - Claude 3.7 Sonnet": "claude-3-7-sonnet-20250219",
    "Anthropic - Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
    "Anthropic - Claude 3.5 Haiku": "claude-3-5-haiku-20241022",
    # Google models
    "Google - Gemini 2.0 Pro": "gemini/gemini-2.0-pro",
    "Google - Gemini 2.0 Flash": "gemini/gemini-2.0-flash",
    "Google - Gemini 2.0 Flash Exp": "gemini/gemini-2.0-flash-exp",
    # Common models
    "default_model_name": "gpt-4o-mini",
}

# Settings constants
SETTINGS_CHAT_MODEL = "chat_model"
SETTINGS_TEMPERATURE = "temperature"
SETTINGS_TOP_P = "top_p"
SETTINGS_SUMMARIZE_SEARCH_RESULTS = "summarize_search_results"
SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING = "use_dynamic_conversation_routing"
SETTINGS_VISION_MODEL = "vision_model"
SETTINGS_ENABLE_TTS_RESPONSE = "enable_tts_response"
SETTINGS_TTS_MODEL = "tts_model"
SETTINGS_TTS_VOICE_PRESET_MODEL = "tts_voice_preset_model"
SETTINGS_IMAGE_GEN_IMAGE_SIZE = "image_gen_image_size"
SETTINGS_IMAGE_GEN_IMAGE_QUALITY = "image_gen_image_quality"
SETTINGS_IMAGE_GEN_BACKGROUND = "image_gen_background"
SETTINGS_IMAGE_GEN_OUTPUT_FORMAT = "image_gen_output_format"
SETTINGS_IMAGE_GEN_OUTPUT_COMPRESSION = "image_gen_output_compression"
SETTINGS_IMAGE_GEN_MODERATION = "image_gen_moderation"
SETTINGS_TRIMMED_MESSAGES = "trimmed_messages"
SETTINGS_WEB_SEARCH_MODEL = "web_search_model"

# Default values for settings
SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE = True
SETTINGS_SUMMARIZE_SEARCH_RESULTS_DEFAULT_VALUE = True
SETTINGS_ENABLE_TTS_RESPONSE_DEFAULT_VALUE = False
SETTINGS_TRIMMED_MESSAGES_DEFAULT_VALUE = True

# Web search models
WEB_SEARCH_MODELS = [
    "gemini/gemini-2.0-flash-lite",
    "openai/gpt-4o",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
]

DEFAULT_WEB_SEARCH_MODEL = "gemini/gemini-2.0-flash-lite"
