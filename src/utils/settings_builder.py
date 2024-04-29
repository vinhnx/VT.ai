from typing import Any, Dict

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch

from utils import llm_config as conf


async def build_settings() -> Dict[str, Any]:
    """
    Builds and sends chat settings to the user for configuration.
    """
    settings = await cl.ChatSettings(
        [
            Select(
                id=conf.SETTINGS_CHAT_MODEL,
                label="Chat Model",
                description="""
                Select the Large Language Model (LLM) you want to use for chat conversations. Different models have varying strengths and capabilities.

                (NOTE) For using Ollama to get up and running with large language models locally. Please refer to quick start guide: https://github.com/ollama/ollama/blob/main/README.md#quickstart""",
                values=conf.MODELS,
                initial_value=conf.DEFAULT_MODEL,
            ),
            Slider(
                id=conf.SETTINGS_TEMPERATURE,
                label="Temperature",
                description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",
                min=0,
                max=2.0,
                step=0.1,
                initial=conf.DEFAULT_TEMPERATURE,
                tooltip="Adjust the temperature parameter",
            ),
            Slider(
                id=conf.SETTINGS_TOP_P,
                label="Top P",
                description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.",
                min=0.1,
                max=1.0,
                step=0.1,
                initial=conf.DEFAULT_TOP_P,
                tooltip="Adjust the top P parameter",
            ),
            Select(
                id=conf.SETTINGS_VISION_MODEL,
                label="Vision Model",
                description="Choose the vision model to analyze and understand images. This enables features like image description and object recognition.",
                values=conf.VISION_MODEL_MODELS,
                initial_value=conf.DEFAULT_VISION_MODEL,
            ),
            Switch(
                id=conf.SETTINGS_ENABLE_TTS_RESPONSE,
                label="Enable TTS",
                description=f"This feature allows you to hear the chat responses spoken aloud, which can be helpful for accessibility or multitasking. Note that this action requires an OpenAI API key. Default value is {conf.SETTINGS_ENABLE_TTS_RESPONSE_DEFAULT_VALUE}.",
                initial=conf.SETTINGS_ENABLE_TTS_RESPONSE_DEFAULT_VALUE,
            ),
            Select(
                id=conf.SETTINGS_TTS_MODEL,
                label="TTS Model",
                description="Select the TTS model to use for generating speech. Different models offer distinct voice styles and characteristics.",
                values=conf.TTS_MODEL_MODELS,
                initial_value=conf.DEFAULT_TTS_MODEL,
            ),
            Select(
                id=conf.SETTINGS_TTS_VOICE_PRESET_MODEL,
                label="TTS - Voice options",
                description="Choose the specific voice preset you prefer for TTS responses. Each preset offers a unique vocal style and tone.",
                values=conf.TTS_VOICE_PRESETS,
                initial_value=conf.DEFAULT_TTS_PRESET,
            ),
            Switch(
                id=conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING,
                label="Use dynamic conversation routing",
                description=f"This experimental feature automatically switches to specialized models based on your input. For example, if you ask to generate an image, it will use an image generation model like DALL·E 3. Note that this action requires an OpenAI API key. Default value is {conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE}",
                initial=conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE,
            ),
            Switch(
                id=conf.SETTINGS_TRIMMED_MESSAGES,
                label="Trimming Input Messages",
                description="Ensure messages does not exceed a model's token limit",
                initial=conf.SETTINGS_TRIMMED_MESSAGES_DEFAULT_VALUE,
            ),
            Select(
                id=conf.SETTINGS_IMAGE_GEN_IMAGE_STYLE,
                label="Image Gen Image Style",
                description="Set the style of the generated images.Vivid : hyper-real and dramatic images. Natural: produce more natural, less hyper-real looking images.",
                values=conf.SETTINGS_IMAGE_GEN_IMAGE_STYLES,
                initial_value=conf.SETTINGS_IMAGE_GEN_IMAGE_STYLES[0],
            ),
            Select(
                id=conf.SETTINGS_IMAGE_GEN_IMAGE_QUALITY,
                label="Image Gen Image Quality",
                description="Set the quality of the image that will be generated. HD creates images with finer details and greater consistency across the image",
                values=conf.SETTINGS_IMAGE_GEN_IMAGE_QUALITIES,
                initial_value=conf.SETTINGS_IMAGE_GEN_IMAGE_QUALITIES[0],
            ),
        ]
    ).send()

    return settings
