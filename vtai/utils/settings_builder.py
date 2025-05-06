"""
Settings builder module for the VT.ai application.

This module provides functions to build and configure user settings for the chat interface.
"""

from typing import Any, Dict, List, Union

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch

from vtai.utils import constants as const
from vtai.utils import llm_providers_config as conf


async def build_settings() -> Dict[str, Any]:
    """
    Builds and sends chat settings to the user for configuration.

    Creates a complete settings panel with all configurable options for the VT.ai
    application, including model selection, conversation parameters, and media settings.

    Returns:
            Dict[str, Any]: The configured user settings as a dictionary
    """
    settings_widgets = _create_settings_widgets()
    settings = await cl.ChatSettings(settings_widgets).send()
    return settings


def _create_settings_widgets() -> List[Union[Select, Slider, Switch]]:
    """
    Creates the list of settings widgets for the chat interface.

    Returns:
            List[Union[Select, Slider, Switch]]: A list of input widgets for the settings panel
    """
    return [
        # ===== CORE CONVERSATION SETTINGS =====
        Select(
            id=conf.SETTINGS_CHAT_MODEL,
            label="Chat Model",
            description="Select the Large Language Model (LLM) for chat conversations. "
            "Different models have varying capabilities and performance characteristics.",
            values=conf.MODELS,
            initial_value=conf.DEFAULT_MODEL,
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
        # ===== REASONING SETTINGS =====
        Select(
            id=conf.SETTINGS_REASONING_EFFORT,
            label="Reasoning Depth",
            description="Control how much reasoning effort models use when thinking through complex problems. "
            "'Low' is faster but less thorough, 'Medium' balances speed and depth, while 'High' provides "
            "the most detailed reasoning but may take longer to generate responses.",
            values=conf.REASONING_EFFORT_LEVELS,
            initial_value=conf.DEFAULT_REASONING_EFFORT,
        ),
    ]
