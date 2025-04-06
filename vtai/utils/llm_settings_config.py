"""
Settings configuration for LLM models and chat profiles.
"""

from typing import Dict, List

import chainlit as cl
from pydantic import BaseModel

# Update imports to use vtai namespace
from vtai.utils.chat_profile import AppChatProfileType


# Define AppChatProfileModel class
class AppChatProfileModel(BaseModel):
    title: str
    description: str
    icon: str
    is_default: bool = False


from chainlit.types import ChatProfile

# Settings keys - grouped for better organization
# Chat settings
SETTINGS_CHAT_MODEL: str = "settings_chat_model"
SETTINGS_TEMPERATURE: str = "settings_temperature"
SETTINGS_TOP_K: str = "settings_top_k"
SETTINGS_TOP_P: str = "settings_top_p"
SETTINGS_TRIMMED_MESSAGES: str = "settings_trimmed_messages"

# Image and Vision settings
SETTINGS_VISION_MODEL: str = "settings_vision_model"
SETTINGS_IMAGE_GEN_LLM_MODEL: str = "settings_image_gen_llm_model"
SETTINGS_IMAGE_GEN_IMAGE_STYLE: str = "settings_image_gen_image_style"
SETTINGS_IMAGE_GEN_IMAGE_QUALITY: str = "settings_image_gen_image_quality"

# Voice and TTS settings
SETTINGS_TTS_MODEL: str = "settings_tts_model"
SETTINGS_TTS_VOICE_PRESET_MODEL: str = "settings_tts_voice_preset_model"
SETTINGS_ENABLE_TTS_RESPONSE: str = "settings_enable_tts_response"

# Routing settings
SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING: str = (
    "settings_use_dynamic_conversation_routing"
)
SETTINGS_USE_THINKING_MODEL: str = "settings_use_thinking_model"

# Default values
DEFAULT_TEMPERATURE: float = 0.8
DEFAULT_TOP_P: float = 1.0
DEFAULT_MODEL: str = "gpt-4o-mini"
DEFAULT_IMAGE_GEN_MODEL: str = "dall-e-3"
DEFAULT_VISION_MODEL: str = "gemini/gemini-2.0-flash"
DEFAULT_TTS_MODEL: str = "gpt-4o-mini-tts"
DEFAULT_TTS_PRESET: str = "nova"
DEFAULT_WHISPER_MODEL: str = "whisper-1"

# Default boolean settings
SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE: bool = True
SETTINGS_TRIMMED_MESSAGES_DEFAULT_VALUE: bool = True
SETTINGS_ENABLE_TTS_RESPONSE_DEFAULT_VALUE: bool = True
SETTINGS_USE_THINKING_MODEL_DEFAULT_VALUE: bool = False

# List of models that benefit from <think> tag for reasoning
REASONING_MODELS = [
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


# Function to check if a model is a reasoning model
def is_reasoning_model(model_id: str) -> bool:
    """
    Check if a model is a reasoning model that benefits from <think> tags.

    Args:
        model_id: The ID of the model to check

    Returns:
        bool: True if the model is a reasoning model, False otherwise
    """
    return any(reasoning_model in model_id for reasoning_model in REASONING_MODELS)


# Image generation options
SETTINGS_IMAGE_GEN_IMAGE_STYLES: List[str] = ["vivid", "natural"]
SETTINGS_IMAGE_GEN_IMAGE_QUALITIES: List[str] = ["standard", "hd"]

# TTS options
TTS_VOICE_PRESETS: List[str] = [
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
]

# Model mappings
TTS_MODELS_MAP: Dict[str, str] = {
    "OpenAI - GPT-4o mini TTS": "gpt-4o-mini-tts",
    "OpenAI - Text-to-speech 1": "tts-1",
    "OpenAI - Text-to-speech 1 HD": "tts-1-hd",
}

IMAGE_GEN_MODELS_ALIAS_MAP: Dict[str, str] = {
    "OpenAI - DALL·E 3": "dall-e-3",
}

VISION_MODEL_MAP: Dict[str, str] = {
    "OpenAI - GPT-4o": "gpt-4o",
    "OpenAI - GPT 4 Turbo": "gpt-4-turbo",
    "Google - Gemini 1.5 Flash": "gemini/gemini-2.0-flash",
    "Google - Gemini 1.5 Pro": "gemini/gemini-2.0-pro",
    "Ollama - LLama 3.2 Vision": "ollama/llama3.2-vision",
}

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
    # OpenRouter models
    "OpenRouter - DeepSeek R1 (free)": "openrouter/deepseek/deepseek-r1:free",
    "OpenRouter - DeepSeek R1": "openrouter/deepseek/deepseek-r1",
    "OpenRouter - DeepSeek V3 0324 (free)": "openrouter/deepseek/deepseek-chat-v3-0324:free",
    "OpenRouter - DeepSeek V3 0324": "openrouter/deepseek/deepseek-chat-v3-0324",
    "OpenRouter - Anthropic: Claude 3.7 Sonnet (thinking)": "openrouter/anthropic/claude-3.7-sonnet:thinking",
    "OpenRouter - Anthropic: Claude 3.7 Sonnet": "openrouter/anthropic/claude-3.7-sonnet",
    "OpenRouter - Google: Gemini 2.5 Pro Experimental (free)": "openrouter/google/gemini-2.5-pro-exp-03-25:free",
    "OpenRouter - Google: Gemini 2.5 Pro Preview": "openrouter/google/gemini-2.5-pro-preview-03-25",
    "OpenRouter - Google: Gemini 2.0 Flash Thinking Experimental (free)": "openrouter/google/gemini-2.0-flash-thinking-exp:free",
    "OpenRouter - Google: Gemini 2.0 Flash Experimental (free)": "openrouter/google/gemini-2.0-flash-exp:free",
    "OpenRouter - Google: Gemma 3 27B (free)": "openrouter/google/gemma-3-27b-it:free",
    "OpenRouter - Meta: Llama 4 Maverick (free)": "openrouter/meta-llama/llama-4-maverick:free",
    "OpenRouter - Meta: Llama 4 Maverick": "openrouter/meta-llama/llama-4-maverick",
    "OpenRouter - Meta: Llama 4 Scout (free)": "openrouter/meta-llama/llama-4-scout:free",
    "OpenRouter - Meta: Llama 4 Scout": "openrouter/meta-llama/llama-4-scout",
    "OpenRouter - Qwen QWQ 32B (free)": "openrouter/qwen/qwq-32b:free",
    "OpenRouter - Qwen QWQ 32B": "openrouter/qwen/qwq-32b",
    "OpenRouter - Qwen 2.5 VL 32B (free)": "openrouter/qwen/qwen2.5-vl-32b-instruct:free",
    "OpenRouter - Qwen 2.5 Coder 32B": "openrouter/qwen/qwen-2.5-coder-32b-instruct",
    "OpenRouter - Mistral: Mistral Small 3.1 24B (free)": "openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
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

# ICONS_PROVIDER_MAP modification - updating resource paths
ICONS_PROVIDER_MAP: Dict[str, str] = {
    # App icons
    "VT.ai": "./vtai/resources/vt.jpg",
    "Mino": "./vtai/resources/vt.jpg",
    # OpenAI icons
    "tts-1": "./vtai/resources/chatgpt-icon.png",
    "tts-1-hd": "./vtai/resources/chatgpt-icon.png",
    "OpenAI": "./vtai/resources/chatgpt-icon.png",
    "dall-e-3": "./vtai/resources/chatgpt-icon.png",
    "gpt-4": "./vtai/resources/chatgpt-icon.png",
    "gpt-4o": "./vtai/resources/chatgpt-icon.png",
    "gpt-4-turbo": "./vtai/resources/chatgpt-icon.png",
    "gpt-3.5-turbo": "./vtai/resources/chatgpt-icon.png",
    # Other provider icons
    "Ollama": "./vtai/resources/ollama.png",
    "Anthropic": "./vtai/resources/claude-ai-icon.png",
    "Google": "./vtai/resources/google-gemini-icon.png",
    "Groq": "./vtai/resources/groq.ico",
    "command": "./vtai/resources/cohere.ico",
    "command-r": "./vtai/resources/cohere.ico",
    "command-light": "./vtai/resources/cohere.ico",
    "command-r-plus": "./vtai/resources/cohere.ico",
    "claude-2": "./vtai/resources/claude-ai-icon.png",
    "claude-3-sonnet-20240229": "./vtai/resources/claude-ai-icon.png",
    "claude-3-haiku-20240307": "./vtai/resources/claude-ai-icon.png",
    "claude-3-opus-20240229": "./vtai/resources/claude-ai-icon.png",
    "groq/llama3-8b-8192": "./vtai/resources/groq.ico",
    "groq/llama3-70b-8192": "./vtai/resources/groq.ico",
    "groq/mixtral-8x7b-32768": "./vtai/resources/groq.ico",
    "gemini/gemini-1.5-pro-latest": "./vtai/resources/google-gemini-icon.png",
    "gemini/gemini-1.5-flash-latest": "./vtai/resources/google-gemini-icon.png",
    "openrouter/mistralai/mistral-7b-instruct": "./vtai/resources/openrouter.ico",
    "OpenRouter - Mistral 7b instruct Free": "./vtai/resources/openrouter.ico",
    "ollama/llama3": "./vtai/resources/ollama.png",
    "ollama/mistral": "./vtai/resources/ollama.png",
}

# Derive lists from dictionaries
NAMES: List[str] = list(MODEL_ALIAS_MAP.keys())
MODELS: List[str] = list(MODEL_ALIAS_MAP.values())

IMAGE_GEN_NAMES: List[str] = list(IMAGE_GEN_MODELS_ALIAS_MAP.keys())
IMAGE_GEN_MODELS: List[str] = list(IMAGE_GEN_MODELS_ALIAS_MAP.values())

VISION_MODEL_NAMES: List[str] = list(VISION_MODEL_MAP.keys())
VISION_MODEL_MODELS: List[str] = list(VISION_MODEL_MAP.values())

TTS_MODEL_NAMES: List[str] = list(TTS_MODELS_MAP.keys())
TTS_MODEL_MODELS: List[str] = list(TTS_MODELS_MAP.values())

# Chat profiles
APP_CHAT_PROFILE_CHAT = AppChatProfileModel(
    title=AppChatProfileType.CHAT.value,
    description="Multi-modal chat with LLM.",
    icon=ICONS_PROVIDER_MAP["Mino"],
    is_default=True,
)

APP_CHAT_PROFILE_ASSISTANT = AppChatProfileModel(
    title=AppChatProfileType.ASSISTANT.value,
    description="[Beta] Use Mino built-in Assistant to ask complex question. Currently support Math Calculator",
    icon=ICONS_PROVIDER_MAP["Mino"],
    is_default=True,
)

APP_CHAT_PROFILES: List[AppChatProfileModel] = [
    APP_CHAT_PROFILE_CHAT,
    APP_CHAT_PROFILE_ASSISTANT,
]

# Update to use markdown_description instead of description for Chainlit v2.0.0
CHAT_PROFILES: List[ChatProfile] = [
    ChatProfile(
        name=profile.title,
        markdown_description=profile.description,
        starters=[
            cl.Starter(
                label="Learn a New Skill",
                message="I want to start learning Python programming. Can you outline the basic concepts I should focus on first and suggest some beginner-friendly online resources?",
            ),
            cl.Starter(
                label="Brainstorm Blog Post Ideas",
                message="I want to write a blog post about sustainable living. Can you help me brainstorm 5 potential article titles and a brief outline for one of them?",
            ),
            cl.Starter(
                label="Plan a Healthy Meal",
                message="Suggest a healthy and balanced dinner recipe that includes chicken and vegetables, and takes less than 45 minutes to prepare. Please list the ingredients and step-by-step instructions.",
            ),
            cl.Starter(
                label="Creative Story Prompt",
                message="Give me a creative writing prompt involving a hidden world discovered through an old bookstore.",
            ),
        ],
    )
    for profile in APP_CHAT_PROFILES
]
