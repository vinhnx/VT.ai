"""
Settings configuration for LLM models and chat profiles.
"""

import random
from random import choice, shuffle
from typing import Dict, List

import chainlit as cl
from pydantic import BaseModel

from vtai.router.trainer import create_routes

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

# Web search settings
SETTINGS_SUMMARIZE_SEARCH_RESULTS: str = "settings_summarize_search_results"

# Image and Vision settings
SETTINGS_VISION_MODEL: str = "settings_vision_model"
SETTINGS_IMAGE_GEN_LLM_MODEL: str = "settings_image_gen_llm_model"
SETTINGS_IMAGE_GEN_IMAGE_STYLE: str = "settings_image_gen_image_style"
SETTINGS_IMAGE_GEN_IMAGE_QUALITY: str = "settings_image_gen_image_quality"
SETTINGS_IMAGE_GEN_IMAGE_SIZE: str = "settings_image_gen_image_size"
SETTINGS_IMAGE_GEN_BACKGROUND: str = "settings_image_gen_background"
SETTINGS_IMAGE_GEN_OUTPUT_FORMAT: str = "settings_image_gen_output_format"
SETTINGS_IMAGE_GEN_MODERATION: str = "settings_image_gen_moderation"
SETTINGS_IMAGE_GEN_OUTPUT_COMPRESSION: str = "settings_image_gen_output_compression"

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
DEFAULT_IMAGE_GEN_MODEL: str = "gpt-image-1"  # Updated from "dall-e-3" to "gpt-image-1"
DEFAULT_VISION_MODEL: str = "gemini/gemini-2.0-flash"
DEFAULT_TTS_MODEL: str = "gpt-4o-mini-tts"
DEFAULT_TTS_PRESET: str = "nova"
DEFAULT_WHISPER_MODEL: str = "whisper-1"
DEFAULT_AUDIO_UNDERSTANDING_MODEL: str = "gpt-4o"

# Default image generation settings
DEFAULT_IMAGE_GEN_BACKGROUND: str = "auto"  # For GPT-Image-1: auto, transparent, opaque
DEFAULT_IMAGE_GEN_OUTPUT_FORMAT: str = "jpeg"  # For GPT-Image-1: png, jpeg, webp
DEFAULT_IMAGE_GEN_MODERATION: str = "low"  # For GPT-Image-1: auto, low
DEFAULT_IMAGE_GEN_OUTPUT_COMPRESSION: int = (
    100  # For GPT-Image-1: 0-100 (webp/jpeg only)
)

# Default boolean settings
SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE: bool = True
SETTINGS_TRIMMED_MESSAGES_DEFAULT_VALUE: bool = True
SETTINGS_ENABLE_TTS_RESPONSE_DEFAULT_VALUE: bool = True
SETTINGS_USE_THINKING_MODEL_DEFAULT_VALUE: bool = False
SETTINGS_SUMMARIZE_SEARCH_RESULTS_DEFAULT_VALUE: bool = True

# List of models that benefit from <think> tag for reasoning
REASONING_MODELS = [
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
    "OpenAI - GPT Image 1": "gpt-image-1",  # Only support GPT-Image-1
}

VISION_MODEL_MAP: Dict[str, str] = {
    "OpenAI - GPT-4o": "gpt-4o",
    "OpenAI - GPT 4 Turbo": "gpt-4-turbo",
    "Google - Gemini 2.0 Flash": "gemini/gemini-2.0-flash",
    "Google - Gemini 2.5 Pro": "gemini/gemini-2.5-pro",
    "Ollama - LLama 3.2 Vision": "ollama/llama3.2-vision",
}

MODEL_ALIAS_MAP: Dict[str, str] = {
    # DeepSeek models
    "DeepSeek R1": "deepseek/deepseek-reasoner",
    "DeepSeek V3": "deepseek/deepseek-chat",
    "DeepSeek Coder": "deepseek/deepseek-coder",
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
    "Google - Gemini 2.0 Pro": "gemini/gemini-2.5-pro",
    "Google - Gemini 2.0 Flash": "gemini/gemini-2.0-flash",
    "Google - Gemini 2.0 Flash Exp": "gemini/gemini-2.5-flash-exp",
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

# Define the icon path constant
ICON_PATH = "./vtai/resources/vt.jpg"

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
    icon=ICON_PATH,
    is_default=True,
)

APP_CHAT_PROFILE_ASSISTANT = AppChatProfileModel(
    title=AppChatProfileType.ASSISTANT.value,
    description="[Beta] Use Mino built-in Assistant to ask complex question. Currently support Math Calculator",
    icon=ICON_PATH,
    is_default=True,
)

APP_CHAT_PROFILES: List[AppChatProfileModel] = [
    APP_CHAT_PROFILE_CHAT,
    APP_CHAT_PROFILE_ASSISTANT,
]

# Starter prompt data and functions
STARTER_PROMPTS = [
    {
        "label": "Generating an image",
        "message": "Generate an image of a futuristic city skyline at sunset with flying cars.",
    },
    {
        "label": "Brainstorm Blog Post Ideas",
        "message": "I want to write a blog post about sustainable living. Can you help me brainstorm 5 potential article titles and a brief outline for one of them?",
    },
    {
        "label": "Plan a Healthy Meal",
        "message": "Suggest a healthy and balanced dinner recipe that includes chicken and vegetables, and takes less than 45 minutes to prepare. Please list the ingredients and step-by-step instructions.",
    },
    {
        "label": "Creative Story Prompt",
        "message": "Give me a creative writing prompt involving a hidden world discovered through an old bookstore.",
    },
    {
        "label": "Code Review Help",
        "message": "I've written a Python function to find prime numbers. Can you review it for efficiency and suggest improvements?\n\n```python\ndef is_prime(n):\n    if n <= 1:\n        return False\n    if n <= 3:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True\n```",
    },
]

# Random label and message generators
CREATIVE_LABELS = [
    "Exploring {topic}",
    "Help with {topic}",
    "Let's discuss {topic}",
    "Ideas for {topic}",
    "Understanding {topic}",
]

TOPICS = [
    "AI ethics",
    "sustainable technology",
    "data visualization",
    "machine learning",
    "creative writing",
    "productivity hacks",
    "coding challenges",
    "future trends",
    "design thinking",
]


def generate_random_prompt():
    """Generate a random prompt with creative label and message"""
    topic = choice(TOPICS)
    label_template = choice(CREATIVE_LABELS)
    label = label_template.format(topic=topic)

    message_templates = [
        f"I'm interested in learning more about {topic}. Can you provide an overview?",
        f"What are the latest developments in {topic}?",
        f"How can I apply {topic} principles in my daily work?",
        f"What are some beginner-friendly resources to learn about {topic}?",
        f"Can you compare different approaches to {topic}?",
    ]

    message = choice(message_templates)
    return {"label": label, "message": message}


# Route-based starter prompts
def build_starters_from_routes(max_count=5):
    """
    Build starter prompts from the router routes.
    Each route category will be converted into a starter prompt with a short label and verbose message.

    Args:
        max_count (int): Maximum number of starters to return

    Returns:
        list: List of cl.Starter objects
    """
    # Get all routes from the router
    routes = create_routes()

    # Create mapping of short labels for each route
    route_labels = {
        "text-processing": "Text Analysis",
        "vision-image-processing": "Analyze Image",
        "casual-conversation": "Chat",
        "image-generation": "Create Image",
        "curious": "Tell Me About",
        "code-assistance": "Code Help",
        "data-analysis": "Analyze Data",
        "creative-writing": "Write Something",
        "planning-organization": "Plan This",
        "troubleshooting": "Fix My Issue",
    }

    # Create expanded messages for each route
    route_messages = {}
    for route in routes:
        if route.name in route_labels:
            # Select a random utterance from the route
            utterance = random.choice(route.utterances)

            # Expand the utterance into a more verbose message
            if route.name == "image-generation":
                route_messages[route.name] = (
                    f"I'd like you to {utterance}. Please make it highly detailed with vibrant colors and an interesting composition."
                )

            elif route.name == "code-assistance":
                route_messages[route.name] = (
                    f"{utterance}. I'm looking for clean, efficient code with good documentation. Please explain your reasoning and any best practices you're applying."
                )

            elif route.name == "data-analysis":
                route_messages[route.name] = (
                    f"{utterance}. I'm interested in both the statistical significance and practical implications of the findings. Please include visual representation suggestions if appropriate."
                )

            elif route.name == "creative-writing":
                route_messages[route.name] = (
                    f"{utterance}. I'd like something unique with vivid imagery and compelling character development. Feel free to explore unexpected directions."
                )

            elif route.name == "planning-organization":
                route_messages[route.name] = (
                    f"{utterance}. I'm looking for a comprehensive approach that considers potential obstacles and includes contingency plans. Please make it practical and implementable."
                )

            elif route.name == "troubleshooting":
                route_messages[route.name] = (
                    f"{utterance}. I've already tried restarting and checking basic connectivity. Please provide a step-by-step diagnostic process and potential solutions ranked by likelihood."
                )

            elif route.name == "vision-image-processing":
                route_messages[route.name] = (
                    f"{utterance}. Please provide details about the key elements, composition, color scheme, and any text or symbols present. Also share any insights about the context or purpose of the image."
                )

            elif route.name == "text-processing":
                route_messages[route.name] = (
                    f"{utterance}. I'd like a thorough analysis that covers tone, key arguments, implicit assumptions, and overall effectiveness. Please suggest improvements where appropriate."
                )

            elif route.name == "casual-conversation":
                route_messages[route.name] = (
                    f"{utterance} I'd love to hear your thoughts on this in a conversational way, as if we're just chatting casually."
                )

            elif route.name == "curious":
                route_messages[route.name] = (
                    f"{utterance} Please provide a comprehensive explanation with interesting facts, historical context, and current developments. I'm particularly interested in aspects that might surprise someone new to the topic."
                )

    # Create starters from routes
    all_starters = []
    route_names = list(route_labels.keys())
    # Shuffle to get random selection each time
    random.shuffle(route_names)

    # Select up to max_count routes
    selected_routes = route_names[:max_count]

    for route_name in selected_routes:
        label = route_labels.get(route_name)
        message = route_messages.get(route_name)

        if label and message:
            all_starters.append({"label": label, "message": message})

    # Convert to Chainlit Starter objects
    return [
        cl.Starter(label=item["label"], message=item["message"])
        for item in all_starters
    ]


def get_shuffled_starters(use_random=False, max_count=5):
    """
    Get shuffled starters for chat profiles

    Args:
        use_random (bool): Whether to use route-based starters (True) or static starters (False)
        max_count (int): Maximum number of starters to return

    Returns:
        list: List of cl.Starter objects
    """
    if use_random:
        # Use dynamic route-based starters
        return build_starters_from_routes(max_count=max_count)
    else:
        # Use static starters
        starters_data = STARTER_PROMPTS.copy()
        shuffle(starters_data)

        # Limit to max_count
        starters_data = starters_data[:max_count]

        # Convert to cl.Starter objects
        return [
            cl.Starter(label=item["label"], message=item["message"])
            for item in starters_data
        ]


# Update to use markdown_description instead of description for Chainlit v2.0.0
CHAT_PROFILES: List[ChatProfile] = [
    ChatProfile(
        name=profile.title,
        markdown_description=profile.description,
        starters=get_shuffled_starters(use_random=True),
    )
    for profile in APP_CHAT_PROFILES
]

"""
Configuration for the LLM providers.
"""

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


def get_provider_config(provider_name):
    """
    Get the configuration for a specific LLM provider.

    Args:
        provider_name (str): The name of the provider (e.g., "openai", "anthropic")

    Returns:
        dict: Configuration for the specified provider
    """
    return DEFAULT_PROVIDERS_CONFIG.get(provider_name, {})
