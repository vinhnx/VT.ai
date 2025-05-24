"""
Settings builder module for the VT.ai application.

This module provides functions to build and configure user settings for the chat interface.
"""

from typing import Any, Dict, List, Union

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch, TextInput
from utils import constants as const
from utils import llm_providers_config as conf

from vtai.utils.api_keys import encrypt_api_key

# NOTE: For public/shared deployments, do NOT put your own API keys in .env.
# Use Chainlit's user_env config to prompt each user for their own API keys (BYOK).
# See: https://docs.chainlit.io/integrations/user-env

DEFAULT_SYSTEM_PROMPT = "You are VT.ai, an helpful AI assistant that helps users solve problems. Be clear, concise, and always provide actionable, step-by-step guidance."


async def build_settings() -> Dict[str, Any]:
    """
    Builds and sends chat settings to the user for configuration.

    Creates a complete settings panel with all configurable options for the VT.ai
    application, including model selection, conversation parameters, and media settings.
    Adds BYOK fields for free users.

    Returns:
            Dict[str, Any]: The configured user settings as a dictionary
    """
    settings_widgets = _create_settings_widgets()

    settings = await cl.ChatSettings(settings_widgets).send()
    return settings


def _create_settings_widgets() -> List[Union[Select, Slider, Switch, TextInput]]:
    """
    Creates the list of settings widgets for the chat interface.

    Returns:
            List[Union[Select, Slider, Switch, TextInput]]: A list of input widgets for the settings panel
    """

    def _encrypted_or_plain(val):
        if val and not val.startswith("gAAAA"):
            return encrypt_api_key(val)
        return val

    def _masked_or_empty(val):
        # Always mask if any value is present (even encrypted)
        if val:
            return "********"
        return ""

    widgets = [
        Select(
            id="show_profile_select",
            label="My Profile",
            description="Select to view your user profile.",
            values=["No", "Yes"],
            initial_value="No",
            on_change="show_user_profile_select",
        ),
        # Add security notice to the description of the first widget (Chat Model)
        Select(
            id=conf.SETTINGS_CHAT_MODEL,
            label="Chat Model",
            description=(
                "**Security Notice:** Your BYOK (Bring Your Own Key) API keys are never stored on the server. "
                "Select the Large Language Model (LLM) for chat conversations. "
                "Different models have varying capabilities and performance characteristics."
            ),
            values=conf.MODELS,
            initial_value=conf.DEFAULT_MODEL,
        ),
        # ===== BYOK FIELDS (masked display, improved description) =====
        TextInput(
            id="byok_openai_api_key",
            label="OpenAI API Key (BYOK)",
            description="Enter your own OpenAI API key. This key is stored securely using strong encryption and never leaves your device.",
            password=True,
            value=_masked_or_empty(cl.user_session.get("byok_openai_api_key")),
        ),
        TextInput(
            id="byok_anthropic_api_key",
            label="Anthropic API Key (BYOK)",
            description="Enter your own Anthropic API key. This key is stored securely using strong encryption and never leaves your device.",
            password=True,
            value=_masked_or_empty(cl.user_session.get("byok_anthropic_api_key")),
        ),
        TextInput(
            id="byok_gemini_api_key",
            label="Gemini API Key (BYOK)",
            description="Enter your own Gemini API key. This key is stored securely using strong encryption and never leaves your device.",
            password=True,
            value=_masked_or_empty(cl.user_session.get("byok_gemini_api_key")),
        ),
        TextInput(
            id="byok_cohere_api_key",
            label="Cohere API Key (BYOK)",
            description="Enter your own Cohere API key. This key is stored securely using strong encryption and never leaves your device.",
            password=True,
            value=_masked_or_empty(cl.user_session.get("byok_cohere_api_key")),
        ),
        TextInput(
            id="byok_mistral_api_key",
            label="Mistral API Key (BYOK)",
            description="Enter your own Mistral API key. This key is stored securely using strong encryption and never leaves your device.",
            password=True,
            value=_masked_or_empty(cl.user_session.get("byok_mistral_api_key")),
        ),
        TextInput(
            id="byok_groq_api_key",
            label="Groq API Key (BYOK)",
            description="Enter your own Groq API key. This key is stored securely using strong encryption and never leaves your device.",
            password=True,
            value=_masked_or_empty(cl.user_session.get("byok_groq_api_key")),
        ),
        TextInput(
            id="byok_deepseek_api_key",
            label="DeepSeek API Key (BYOK)",
            description="Enter your own DeepSeek API key. This key is stored securely using strong encryption and never leaves your device.",
            password=True,
            value=_masked_or_empty(cl.user_session.get("byok_deepseek_api_key")),
        ),
        TextInput(
            id="byok_openrouter_api_key",
            label="OpenRouter API Key (BYOK)",
            description="Enter your own OpenRouter API key. This key is stored securely using strong encryption and never leaves your device.",
            password=True,
            value=_masked_or_empty(cl.user_session.get("byok_openrouter_api_key")),
        ),
        TextInput(
            id="custom_system_prompt",
            label="Custom System Prompt",
            description=(
                "(Optional) Set your own system prompt to guide the AI's behavior. "
                "If left blank, VT.ai will use its default expert assistant prompt."
            ),
            password=False,
            value=cl.user_session.get("custom_system_prompt") or DEFAULT_SYSTEM_PROMPT,
            placeholder=DEFAULT_SYSTEM_PROMPT,
        ),
        TextInput(
            id="ollama_model_name",
            label="Ollama Model Name",
            description="Enter the model name for your local Ollama instance (e.g. 'llama3', 'deepseek-r1:7b', etc).",
            password=False,
            value=cl.user_session.get("ollama_model_name") or "",
            placeholder="llama3",
        ),
        TextInput(
            id="ollama_api_base",
            label="Ollama API Base URL",
            description="(Optional) Set the base URL for your Ollama server (e.g. http://localhost:11434). Leave blank for default.",
            password=False,
            value=cl.user_session.get("ollama_api_base") or "",
            placeholder="http://localhost:11434",
        ),
        TextInput(
            id="lmstudio_model_name",
            label="LM Studio Model Name",
            description="Enter the model name for your local LM Studio instance (e.g. 'phi3', 'llama3', etc).",
            password=False,
            value=cl.user_session.get("lmstudio_model_name") or "",
            placeholder="phi3",
        ),
        TextInput(
            id="lmstudio_api_base",
            label="LM Studio API Base URL",
            description="(Optional) Set the base URL for your LM Studio server (e.g. http://localhost:1234). Leave blank for default.",
            password=False,
            value=cl.user_session.get("lmstudio_api_base") or "",
            placeholder="http://localhost:1234",
        ),
        TextInput(
            id="llamacpp_model_name",
            label="llama.cpp Model Name",
            description="Enter the model name for your local llama.cpp instance (e.g. 'llama-3', 'phi3', etc).",
            password=False,
            value=cl.user_session.get("llamacpp_model_name") or "",
            placeholder="llama-3",
        ),
        TextInput(
            id="llamacpp_api_base",
            label="llama.cpp API Base URL",
            description="(Optional) Set the base URL for your llama.cpp server (e.g. http://localhost:8080). Leave blank for default.",
            password=False,
            value=cl.user_session.get("llamacpp_api_base") or "",
            placeholder="http://localhost:8080",
        ),
        Slider(
            id=conf.SETTINGS_TEMPERATURE,
            label="Temperature",
            description="Controls randomness in responses. Higher values (0.8+) increase creativity "
            "but may reduce accuracy. Lower values (0.2-0.4) produce more focused and deterministic responses.",
            min=0,
            max=2.0,
            step=0.1,
            initial=conf.DEFAULT_TEMPERATURE,
            tooltip="Adjust creativity level",
        ),
        Slider(
            id=conf.SETTINGS_TOP_P,
            label="Top P (Nucleus Sampling)",
            description="Controls response diversity by limiting token selection to the most likely options. "
            "Lower values (0.1-0.3) increase focus on highly probable tokens. "
            "Higher values consider more options.",
            min=0.1,
            max=1.0,
            step=0.1,
            initial=conf.DEFAULT_TOP_P,
            tooltip="Adjust response diversity",
        ),
        Switch(
            id=conf.SETTINGS_TRIMMED_MESSAGES,
            label="Smart Message Trimming",
            description="Automatically manage conversation history to prevent exceeding model token limits. "
            "Keeps the most relevant context within the model's processing capacity.",
            initial=conf.SETTINGS_TRIMMED_MESSAGES_DEFAULT_VALUE,
        ),
        Switch(
            id=conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING,
            label="Dynamic Conversation Routing",
            description="Automatically route queries to specialized models based on content type. "
            "For example, image generation prompts will use GPT-Image-1, and visual analysis "
            "will use vision-capable models.",
            initial=conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE,
        ),
        # ===== VISION & MEDIA SETTINGS =====
        Select(
            id=conf.SETTINGS_VISION_MODEL,
            label="Vision Model",
            description="Select the model for analyzing and understanding images. Vision models can "
            "describe image content, identify objects, and answer questions about visual information.",
            values=conf.VISION_MODEL_MODELS,
            initial_value=conf.DEFAULT_VISION_MODEL,
        ),
        # ===== IMAGE GENERATION SETTINGS =====
        Select(
            id=conf.SETTINGS_IMAGE_GEN_IMAGE_SIZE,
            label="Image Size",
            description="Set output dimensions for generated images. 'Auto' lets the model choose based on your prompt. "
            "Square (1024×1024) works well for most content, while landscape or portrait options "
            "are better for specific composition needs.",
            values=conf.SETTINGS_IMAGE_GEN_IMAGE_SIZES_GPT,
            initial_value=conf.SETTINGS_IMAGE_GEN_IMAGE_SIZES_GPT[0],
        ),
        Select(
            id=conf.SETTINGS_IMAGE_GEN_IMAGE_QUALITY,
            label="Image Quality",
            description="Control the quality level of generated images. Higher settings produce more detailed results "
            "but take longer to generate. Use 'Low' for drafts and 'High' for polished final images.",
            values=conf.SETTINGS_IMAGE_GEN_IMAGE_QUALITIES_GPT,
            initial_value=conf.SETTINGS_IMAGE_GEN_IMAGE_QUALITIES_GPT[0],
        ),
        Select(
            id=conf.SETTINGS_IMAGE_GEN_BACKGROUND,
            label="Background Style",
            description="Set the background type for generated images. 'Transparent' is useful for PNG images "
            "that will be composited with other elements. 'Opaque' provides a solid background.",
            values=conf.SETTINGS_IMAGE_GEN_BACKGROUNDS,
            initial_value=conf.DEFAULT_IMAGE_GEN_BACKGROUND,
        ),
        Select(
            id=conf.SETTINGS_IMAGE_GEN_OUTPUT_FORMAT,
            label="Output Format",
            description="Choose the file format for generated images. PNG supports transparency and is lossless. "
            "JPEG offers smaller file sizes ideal for photographs. WebP balances quality and compression.",
            values=conf.SETTINGS_IMAGE_GEN_OUTPUT_FORMATS,
            initial_value=conf.DEFAULT_IMAGE_GEN_OUTPUT_FORMAT,
        ),
        Slider(
            id=conf.SETTINGS_IMAGE_GEN_OUTPUT_COMPRESSION,
            label="Compression Level",
            description="For JPEG and WebP formats, this controls the compression level. Higher values (100) maintain "
            "maximum quality with larger file sizes. Lower values reduce file size with some quality loss.",
            min=10,
            max=100,
            step=10,
            initial=conf.DEFAULT_IMAGE_GEN_OUTPUT_COMPRESSION,
            tooltip="JPEG/WebP compression (PNG ignores this setting)",
        ),
        Select(
            id=conf.SETTINGS_IMAGE_GEN_MODERATION,
            label="Content Filter",
            description="Set the content moderation strictness. 'Low' allows more creative freedom while "
            "still filtering inappropriate content. 'Auto' applies standard OpenAI content filtering.",
            values=conf.SETTINGS_IMAGE_GEN_MODERATION_LEVELS,
            initial_value=conf.DEFAULT_IMAGE_GEN_MODERATION,
        ),
        # ===== SPEECH & AUDIO SETTINGS =====
        Switch(
            id=conf.SETTINGS_ENABLE_TTS_RESPONSE,
            label="Enable Text-to-Speech",
            description="Convert AI responses to spoken audio. Useful for accessibility, multitasking, "
            "or when you prefer listening to responses rather than reading them.",
            initial=conf.SETTINGS_ENABLE_TTS_RESPONSE_DEFAULT_VALUE,
        ),
        Select(
            id=conf.SETTINGS_TTS_MODEL,
            label="TTS Model",
            description="Select the text-to-speech model to use. Different models offer varying levels "
            "of naturalness, speed, and resource requirements.",
            values=conf.TTS_MODEL_MODELS,
            initial_value=conf.DEFAULT_TTS_MODEL,
        ),
        Select(
            id=conf.SETTINGS_TTS_VOICE_PRESET_MODEL,
            label="Voice Style",
            description="Choose the voice characteristic for text-to-speech output. Each option provides "
            "a different vocal style, tone, and personality.",
            values=conf.TTS_VOICE_PRESETS,
            initial_value=conf.DEFAULT_TTS_PRESET,
        ),
        # ===== WEB SEARCH SETTINGS =====
        Switch(
            id=conf.SETTINGS_SUMMARIZE_SEARCH_RESULTS,
            label="Summarize Search Results",
            description="Generate coherent summaries from web search results instead of showing raw data. "
            "This creates more integrated and readable responses to web search queries.",
            initial=conf.SETTINGS_SUMMARIZE_SEARCH_RESULTS_DEFAULT_VALUE,
        ),
        Select(
            id=const.SETTINGS_WEB_SEARCH_MODEL,
            label="Search Model",
            description="Select the model used for processing and summarizing web search results. "
            "More capable models can better synthesize information from multiple sources.",
            values=const.WEB_SEARCH_MODELS,
            initial_value=const.DEFAULT_WEB_SEARCH_MODEL,
        ),
    ]

    return widgets
