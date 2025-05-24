"""
Settings configuration for LLM models and chat profiles.

This module defines configuration constants and settings for various LLM providers,
models, and feature-specific settings used throughout the VT.ai application.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import chainlit as cl
from pydantic import BaseModel

from .chat_profile import AppChatProfileType
from .starter_prompts import get_shuffled_starters

logger = logging.getLogger(__name__)

# ===== MODEL CLASSES =====


class AppChatProfileModel(BaseModel):
    """
    Defines a chat profile configuration for the application.

    Attributes:
            title: Display name of the chat profile
            description: Text description of the profile's purpose
            icon: Path to the profile icon image
            is_default: Whether this is the default selected profile
    """

    title: str
    description: str
    icon: str
    is_default: bool = False


from chainlit.types import ChatProfile

# ===== SETTING KEY CONSTANTS =====

# Core Model Settings
SETTINGS_CHAT_MODEL: str = "settings_chat_model"
SETTINGS_TEMPERATURE: str = "settings_temperature"
SETTINGS_TOP_P: str = "settings_top_p"
SETTINGS_TOP_K: str = "settings_top_k"  # Not currently exposed in UI
SETTINGS_TRIMMED_MESSAGES: str = "settings_trimmed_messages"
SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING: str = (
    "settings_use_dynamic_conversation_routing"
)
SETTINGS_USE_THINKING_MODEL: str = (
    "settings_use_thinking_model"  # Not currently exposed in UI
)

# Vision Settings
SETTINGS_VISION_MODEL: str = "settings_vision_model"

# Image Generation Settings
SETTINGS_IMAGE_GEN_LLM_MODEL: str = "settings_image_gen_llm_model"
SETTINGS_IMAGE_GEN_IMAGE_STYLE: str = "settings_image_gen_image_style"
SETTINGS_IMAGE_GEN_IMAGE_QUALITY: str = "settings_image_gen_image_quality"
SETTINGS_IMAGE_GEN_IMAGE_SIZE: str = "settings_image_gen_image_size"
SETTINGS_IMAGE_GEN_BACKGROUND: str = "settings_image_gen_background"
SETTINGS_IMAGE_GEN_OUTPUT_FORMAT: str = "settings_image_gen_output_format"
SETTINGS_IMAGE_GEN_MODERATION: str = "settings_image_gen_moderation"
SETTINGS_IMAGE_GEN_OUTPUT_COMPRESSION: str = "settings_image_gen_output_compression"

# Speech & Audio Settings
SETTINGS_TTS_MODEL: str = "settings_tts_model"
SETTINGS_TTS_VOICE_PRESET_MODEL: str = "settings_tts_voice_preset_model"
SETTINGS_ENABLE_TTS_RESPONSE: str = "settings_enable_tts_response"

# Web Search Settings
SETTINGS_SUMMARIZE_SEARCH_RESULTS: str = "settings_summarize_search_results"


# ===== DEFAULT VALUES =====

# Core Model Defaults
DEFAULT_TEMPERATURE: float = 0.8
DEFAULT_TOP_P: float = 1.0
DEFAULT_MODEL: str = "openrouter/google/gemma-3-27b-it:free"

# Vision Model Defaults
DEFAULT_VISION_MODEL: str = "gemini/gemini-2.0-flash"

# Image Generation Defaults
DEFAULT_IMAGE_GEN_MODEL: str = "gpt-image-1"
DEFAULT_IMAGE_GEN_BACKGROUND: str = "auto"
DEFAULT_IMAGE_GEN_OUTPUT_FORMAT: str = "jpeg"
DEFAULT_IMAGE_GEN_MODERATION: str = "low"
DEFAULT_IMAGE_GEN_OUTPUT_COMPRESSION: int = 100

# Speech & Audio Defaults
DEFAULT_TTS_MODEL: str = "gpt-4o-mini-tts"
DEFAULT_TTS_PRESET: str = "nova"
DEFAULT_WHISPER_MODEL: str = "whisper-1"
DEFAULT_AUDIO_UNDERSTANDING_MODEL: str = "gpt-4o"

# Default boolean settings
SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE: bool = True
SETTINGS_TRIMMED_MESSAGES_DEFAULT_VALUE: bool = True
SETTINGS_ENABLE_TTS_RESPONSE_DEFAULT_VALUE: bool = True
SETTINGS_USE_THINKING_MODEL_DEFAULT_VALUE: bool = False
SETTINGS_SUMMARIZE_SEARCH_RESULTS_DEFAULT_VALUE: bool = True


# ===== OPTION LISTS =====

# Image Generation Options
SETTINGS_IMAGE_GEN_IMAGE_STYLES: List[str] = ["vivid", "natural"]
SETTINGS_IMAGE_GEN_IMAGE_QUALITIES: List[str] = ["standard", "hd"]
SETTINGS_IMAGE_GEN_IMAGE_QUALITIES_GPT: List[str] = ["auto", "high", "medium", "low"]
SETTINGS_IMAGE_GEN_IMAGE_SIZES: List[str] = ["1024x1024", "1792x1024", "1024x1792"]
SETTINGS_IMAGE_GEN_IMAGE_SIZES_GPT: List[str] = [
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
]
SETTINGS_IMAGE_GEN_BACKGROUNDS: List[str] = ["auto", "transparent", "opaque"]
SETTINGS_IMAGE_GEN_OUTPUT_FORMATS: List[str] = ["png", "jpeg", "webp"]
SETTINGS_IMAGE_GEN_MODERATION_LEVELS: List[str] = ["auto", "low"]

# Speech & Audio Options
TTS_VOICE_PRESETS: List[str] = [
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
]


# ===== MODEL MAPPINGS =====

# Text-to-Speech Models
TTS_MODELS_MAP: Dict[str, str] = {
    "OpenAI - GPT-4o mini TTS": "gpt-4o-mini-tts",
    "OpenAI - Text-to-speech 1": "tts-1",
    "OpenAI - Text-to-speech 1 HD": "tts-1-hd",
}

# Image Generation Models
IMAGE_GEN_MODELS_ALIAS_MAP: Dict[str, str] = {
    "OpenAI - GPT Image 1": "gpt-image-1",  # Only support GPT-Image-1
}

# Vision Models
VISION_MODEL_MAP: Dict[str, str] = {
    "OpenAI - GPT-4o": "gpt-4o",
    "OpenAI - GPT 4 Turbo": "gpt-4-turbo",
    "Google - Gemini 2.0 Flash": "gemini/gemini-2.0-flash",
    "Google - Gemini 2.5 Pro": "gemini/gemini-2.5-pro",
    "Ollama - LLama 3.2 Vision": "ollama/llama3.2-vision",
}

# Chat Models
MODEL_ALIAS_MAP: Dict[str, str] = {
    # OpenAI models
    "OpenAI - GPT-4.1": "gpt-4.1",
    "OpenAI - GPT-4.1 Mini": "gpt-4.1-mini",
    "OpenAI - GPT-4.1 Nano": "gpt-4.1-nano",
    "OpenAI - GPT-4o Mini": "gpt-4o-mini",
    "OpenAI - GPT-4o": "gpt-4o",
    "OpenAI - GPT-o1": "o1",
    "OpenAI - GPT-4.5 Preview": "gpt-4.5-preview",
    "OpenAI - GPT-o3 Mini": "o3-mini",
    "OpenAI - GPT-o1 Mini": "o1-mini",
    "OpenAI - GPT-o1 Pro": "o1-pro",
    # Anthropic models
    "Anthropic - Claude 4 Sonnet": "claude-sonnet-4-20250514",
    "Anthropic - Claude 4 Opus": "claude-opus-4-20250514",
    "Anthropic - Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
    "Anthropic - Claude 3.5 Haiku": "claude-3-5-haiku-20241022",
    # Google models
    "Google - Gemini 2.5 Pro": "gemini/gemini-2.5-pro-exp-03-25",
    "Google - Gemini 2.5 Flash Preview": "gemini/gemini-2.5-flash-preview-04-17",
    "Google - Gemini 2.0 Flash": "gemini/gemini-2.0-flash",
    "Google - Gemini 2.0 Flash-Lite": "gemini/gemini-2.0-flash-lite",
    # DeepSeek models
    "DeepSeek R1": "deepseek/deepseek-reasoner",
    "DeepSeek V3": "deepseek/deepseek-chat",
    "DeepSeek Coder": "deepseek/deepseek-coder",
    # OpenRouter models
    "OpenRouter - Mistral nemo (free)": "openrouter/mistralai/mistral-nemo:free",
    "OpenRouter - Gemma 3": "openrouter/google/gemma-3-27b-it:free",
    "OpenRouter - Google Gemini 2.0 Flash (free)": "openrouter/google/gemini-2.0-flash-exp:free",
    "OpenRouter - Qwen: Qwen3 0.6B (free)": "openrouter/mistralai/mistral-nemo:free",
    "OpenRouter - DeepSeek R1 (free)": "openrouter/deepseek/deepseek-r1:free",
    "OpenRouter - DeepSeek R1": "openrouter/deepseek/deepseek-r1",
    "OpenRouter - DeepSeek V3 0324 (free)": "openrouter/deepseek/deepseek-chat-v3-0324:free",
    "OpenRouter - DeepSeek V3 0324": "openrouter/deepseek/deepseek-chat-v3-0324",
    "OpenRouter - Anthropic: Claude 3.7 Sonnet (thinking)": "openrouter/anthropic/claude-3.7-sonnet:thinking",
    "OpenRouter - Anthropic: Claude 3.7 Sonnet": "openrouter/anthropic/claude-3.7-sonnet",
    "OpenRouter - Google: Gemini 2.5 Pro Preview": "openrouter/google/gemini-2.5-pro-preview-03-25",
    "OpenRouter - Meta: Llama 4 Maverick": "openrouter/meta-llama/llama-4-maverick",
    "OpenRouter - Meta: Llama 4 Scout": "openrouter/meta-llama/llama-4-scout",
    "OpenRouter - Qwen QWQ 32B": "openrouter/qwen/qwq-32b",
    "OpenRouter - Qwen 2.5 Coder 32B": "openrouter/qwen/qwen-2.5-coder-32b-instruct",
    "OpenRouter - Mistral: Mistral Small 3.1 24B": "openrouter/mistralai/mistral-small-3.1-24b-instruct",
    # Ollama models
    "Ollama - Deepseek R1 1.5B": "ollama/deepseek-r1:1.5b",
    "Ollama - Deepseek R1 7B": "ollama/deepseek-r1:7b",
    "Ollama - Deepseek R1 8B": "ollama/deepseek-r1:8b",
    "Ollama - Deepseek R1 14B": "ollama/deepseek-r1:14b",
    "Ollama - Deepseek R1 32B": "ollama/deepseek-r1:32b",
    "Ollama - Deepseek R1 70B": "ollama/deepseek-r1:70b",
    "Ollama - Qwen2.5-coder 7b": "ollama/qwen2.5-coder",
    "Ollama - Qwen2.5-coder 14b": "ollama/qwen2.5-coder:14b",
    "Ollama - Qwen2.5-coder 32b": "ollama/qwen2.5-coder:32b",
    "Ollama - LLama 3.2 Vision": "ollama/llama3.2-vision",
    "Ollama - LLama 3": "ollama/llama3",
    "Ollama - LLama 3.1": "ollama/llama3.1",
    "Ollama - LLama 3.2 - Mini": "ollama/llama3.2",
    "Ollama - Phi-3": "ollama/phi3",
    "Ollama - Command R": "ollama/command-r",
    "Ollama - Command R+": "ollama/command-r-plus",
    "Ollama - Mistral 7B Instruct": "ollama/mistral",
    "Ollama - Mixtral 8x7B Instruct": "ollama/mixtral",
    # Mistral
    "Mistral Small": "mistral/mistral-small-latest",
    "Mistral Large": "mistral/mistral-large-latest",
    # Groq models
    "Groq - Llama 4 Scout 17b Instruct": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Groq - Llama 3 8b": "groq/llama3-8b-8192",
    "Groq - Llama 3 70b": "groq/llama3-70b-8192",
    "Groq - Mixtral 8x7b": "groq/mixtral-8x7b-32768",
    # Cohere models
    "Cohere - Command": "command",
    "Cohere - Command-R": "command-r",
    "Cohere - Command-Light": "command-light",
    "Cohere - Command-R-Plus": "command-r-plus",
}


# ===== UTILITY FUNCTIONS =====

# Mapping from provider name to its primary API key environment variable name
PROVIDER_TO_KEY_ENV_VAR: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",  # Google Gemini
    "cohere": "COHERE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",  # DeepSeek API
    # Azure is special and handled separately in get_llm_params
}

# Azure specific key environment variable names and their corresponding param names for LiteLLM
AZURE_KEY_MAPPINGS: List[Tuple[str, str]] = [
    ("AZURE_API_KEY", "api_key"),
    ("AZURE_API_BASE", "api_base"),
    ("AZURE_API_VERSION", "api_version"),
]


def get_llm_params(
    model_name: str, user_keys: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Constructs the parameters for a LiteLLM API call, including API key management.
    Priority: user_env (from config.toml UI) > user_keys (from Chat Settings) > os.getenv()
    """
    params: Dict[str, Any] = {}
    user_env = cl.user_session.get("env")  # From config.toml UI
    provider = model_name.split("/")[0] if "/" in model_name else "openai"

    if provider == "azure":
        for azure_env_var, lite_llm_param_name in AZURE_KEY_MAPPINGS:
            key_val_part: Optional[str] = None

            # 1. Try user_env (from config.toml UI)
            if isinstance(user_env, dict) and user_env.get(azure_env_var):
                key_val_part = user_env.get(azure_env_var)
            # 2. Try user_keys (from Chat Settings UI)
            elif (
                isinstance(user_keys, dict)
                and isinstance(user_keys.get("azure"), dict)
                and user_keys["azure"].get(azure_env_var)
            ):
                key_val_part = user_keys["azure"].get(azure_env_var)
            # 3. Try os.getenv()
            elif os.getenv(azure_env_var):
                key_val_part = os.getenv(azure_env_var)

            if key_val_part:  # Only add to params if a non-empty value was found
                params[lite_llm_param_name] = key_val_part

    elif provider in PROVIDER_TO_KEY_ENV_VAR:
        env_var_name = PROVIDER_TO_KEY_ENV_VAR[provider]
        key_val: Optional[str] = None

        # 1. Try user_env (from config.toml UI)
        if isinstance(user_env, dict) and user_env.get(env_var_name):
            key_val = user_env.get(env_var_name)
        # 2. Try user_keys (from Chat Settings UI, keyed by provider name, value is the string key)
        elif (
            isinstance(user_keys, dict)
            and isinstance(user_keys.get(provider), str)
            and user_keys.get(provider)
        ):  # Check if it's a non-empty string
            key_val = user_keys.get(provider)
        # 3. Try os.getenv()
        elif os.getenv(env_var_name):
            key_val = os.getenv(env_var_name)

        if key_val:  # Only add to params if a non-empty value was found
            params["api_key"] = key_val

    return params


# ===== RESOURCE PATHS =====

# Define the icon path constant
ICON_PATH = "./vtai/resources/vt.jpg"


# ===== DERIVED LISTS =====

# Generate lists from model dictionaries for UI display
NAMES: List[str] = list(MODEL_ALIAS_MAP.keys())
MODELS: List[str] = list(MODEL_ALIAS_MAP.values())

IMAGE_GEN_NAMES: List[str] = list(IMAGE_GEN_MODELS_ALIAS_MAP.keys())
IMAGE_GEN_MODELS: List[str] = list(IMAGE_GEN_MODELS_ALIAS_MAP.values())

VISION_MODEL_NAMES: List[str] = list(VISION_MODEL_MAP.keys())
VISION_MODEL_MODELS: List[str] = list(VISION_MODEL_MAP.values())

TTS_MODEL_NAMES: List[str] = list(TTS_MODELS_MAP.keys())
TTS_MODEL_MODELS: List[str] = list(TTS_MODELS_MAP.values())


# ===== CHAT PROFILES =====

# Define application chat profile
APP_CHAT_PROFILE_CHAT = AppChatProfileModel(
    title=AppChatProfileType.CHAT.value,
    description="Multi-modal chat with LLM.",
    icon=ICON_PATH,
    is_default=True,
)

APP_CHAT_PROFILES: List[AppChatProfileModel] = [
    APP_CHAT_PROFILE_CHAT,
]

# Create Chainlit chat profiles from application profiles
CHAT_PROFILES: List[ChatProfile] = [
    ChatProfile(
        name=profile.title,
        markdown_description=profile.description,
        starters=get_shuffled_starters(max_count=5, use_llm=True),
    )
    for profile in APP_CHAT_PROFILES
]

# NOTE: All API key handling in this file is via get_llm_params, which expects decrypted keys only.
# No plain text API keys are stored or handled here. All encryption/decryption is handled upstream.
