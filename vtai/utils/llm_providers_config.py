"""
Settings configuration for LLM models and chat profiles.

This module defines configuration constants and settings for various LLM providers,
models, and feature-specific settings used throughout the VT.ai application.
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path
from random import choice, shuffle
from typing import Any, Dict, List, Optional

import chainlit as cl
from litellm import completion
from pydantic import BaseModel

from vtai.router.trainer import create_routes
from vtai.utils.chat_profile import AppChatProfileType
from vtai.utils.starter_prompts import get_shuffled_starters

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

# Reasoning Settings
SETTINGS_REASONING_EFFORT: str = "settings_reasoning_effort"


# ===== DEFAULT VALUES =====

# Core Model Defaults
DEFAULT_TEMPERATURE: float = 0.8
DEFAULT_TOP_P: float = 1.0
DEFAULT_MODEL: str = "gemini/gemini-2.0-flash"

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

# Default reasoning settings
DEFAULT_REASONING_EFFORT: str = "medium"


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

# Reasoning Options
REASONING_EFFORT_LEVELS: List[str] = ["low", "medium", "high"]

# Models that benefit from <think> tag for reasoning
REASONING_MODELS: List[str] = [
    "deepseek/deepseek-reasoner",
    "openrouter/deepseek/deepseek-r1:free",
    "openrouter/deepseek/deepseek-r1",
    "openrouter/deepseek/deepseek-chat-v3-0324:free",
    "openrouter/deepseek/deepseek-chat-v3-0324",
    "ollama/deepseek-r1:1.5b",
    "ollama/deepseek-r1:7b",
    "ollama/deepseek-r1:8b",
    "ollama/deepseek-r1:14b",
    "ollama/deepseek-r1:32b",
    "ollama/deepseek-r1:70b",
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
    "Anthropic - Claude 3.7 Sonnet": "claude-3-7-sonnet-20250219",
    "Anthropic - Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
    "Anthropic - Claude 3.5 Haiku": "claude-3-5-haiku-20241022",
    # Google models
    "Google - Gemini 2.5 Pro": "gemini/gemini-2.5-pro-exp-03-25",
    "Google - Gemini 2.0 Flash": "gemini/gemini-2.0-flash",
    "Google - Gemini 2.0 Flash-Lite": "gemini/gemini-2.0-flash-lite",
    "Google - Gemini 2.5 Flash Preview": "gemini/gemini-2.5-flash-preview-04-17",
    # DeepSeek models
    "DeepSeek R1": "deepseek/deepseek-reasoner",
    "DeepSeek V3": "deepseek/deepseek-chat",
    "DeepSeek Coder": "deepseek/deepseek-coder",
    # OpenRouter models
    "OpenRouter - Qwen: Qwen3 0.6B (free)": "openrouter/qwen/qwen3-0.6b-04-28:free",
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


def is_reasoning_model(model_id: str) -> bool:
    """
    Check if a model is a reasoning model that benefits from <think> tags.

    Args:
            model_id: The ID of the model to check

    Returns:
            bool: True if the model is a reasoning model, False otherwise
    """
    return any(reasoning_model in model_id for reasoning_model in REASONING_MODELS)


def supports_reasoning(model_id: str) -> bool:
    """
    Check if a model supports LiteLLM's standardized reasoning capabilities.

    This function checks if a model supports the standardized reasoning_content
    feature in LiteLLM, which works across multiple providers including Anthropic,
    DeepSeek, Bedrock, Vertex AI, OpenRouter, XAI, and Google AI.

    Args:
        model_id: The ID of the model to check

    Returns:
        bool: True if the model supports LiteLLM's reasoning capabilities, False otherwise
    """
    try:
        import litellm

        return litellm.supports_reasoning(model=model_id)
    except (ImportError, Exception):
        # Fall back to string matching if litellm.supports_reasoning is not available
        reasoning_providers = [
            "anthropic/",
            "deepseek/",
            "bedrock/",
            "vertexai/",
            "vertex_ai/",
            "openrouter/anthropic/",
            "openrouter/deepseek/",
            "xai/",
            "google/",
        ]
        return any(provider in model_id for provider in reasoning_providers)


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


# ===== PROVIDER CONFIGURATIONS =====

# Default configuration for different LLM providers
DEFAULT_PROVIDERS_CONFIG = {
    "openai": {
        "model": "gpt-4.1-nano",
        "temperature": 0.7,
        "max_tokens": 2000,
    },
    "anthropic": {
        "model": "claude-3-opus-20240229",
        "temperature": 0.7,
        "max_tokens": 2000,
    },
}


def get_provider_config(provider_name: str) -> Dict[str, Any]:
    """
    Get the configuration for a specific LLM provider.

    Args:
            provider_name: The name of the provider (e.g., "openai", "anthropic")

    Returns:
            Dict[str, Any]: Configuration for the specified provider
    """
    return DEFAULT_PROVIDERS_CONFIG.get(provider_name, {})
