"""
Media processing utilities for VT.ai application.

Handles audio transcription, text-to-speech, and image processing.
"""

import os
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chainlit as cl
import litellm
from openai import AsyncOpenAI, OpenAI

from utils import llm_settings_config as conf
from utils.config import temp_dir, logger
from utils.user_session_helper import get_setting, get_user_session_id, update_message_history_from_assistant


async def handle_tts_response(context: str, openai_client: OpenAI) -> None:
    """
    Generates and sends a TTS audio response using OpenAI's Audio API.
    
    Args:
        context: Text to convert to speech
        openai_client: OpenAI client instance
    """
    enable_tts_response = get_setting(conf.SETTINGS_ENABLE_TTS_RESPONSE)
    if enable_tts_response is False or not context:
        return

    model = get_setting(conf.SETTINGS_TTS_MODEL)
    voice = get_setting(conf.SETTINGS_TTS_VOICE_PRESET_MODEL)

    try:
        with openai_client.audio.speech.with_streaming_response.create(
            model=model, voice=voice, input=context
        ) as response:
            temp_filepath = os.path.join(temp_dir.name, "tts-output.mp3")
            response.stream_to_file(temp_filepath)

            await cl.Message(
                author=model,
                content="",
                elements=[
                    cl.Audio(path=temp_filepath, display="inline"),
                    cl.Text(
                        content=f"You're hearing an AI voice generated by OpenAI's {model} model, using the {voice} style. You can customize this in Settings if you'd like!",
                        display="inline",
                    ),
                ],
            ).send()

            update_message_history_from_assistant(context)
    except Exception as e:
        logger.error(f"Error generating TTS response: {e}")
        await cl.Message(content=f"Failed to generate speech: {str(e)}").send()


async def handle_audio_transcribe(path: str, audio_file: Path, async_openai_client: AsyncOpenAI) -> str:
    """
    Transcribes audio to text using OpenAI's Whisper model.
    
    Args:
        path: Path to the audio file
        audio_file: Path object for the audio file
        async_openai_client: AsyncOpenAI client instance
        
    Returns:
        The transcribed text
    """
    model = conf.DEFAULT_WHISPER_MODEL
    try:
        transcription = await async_openai_client.audio.transcriptions.create(
            model=model, file=audio_file
        )
        text = transcription.text

        await cl.Message(
            content="",
            author=model,
            elements=[
                cl.Audio(path=path, display="inline"),
                cl.Text(content=text, display="inline"),
            ],
        ).send()

        update_message_history_from_assistant(text)
        return text
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        await cl.Message(content=f"Failed to transcribe audio: {str(e)}").send()
        return ""


async def handle_vision(
    input_image: str,
    prompt: str,
    is_local: bool = False,
) -> None:
    """
    Handles vision processing tasks using the specified vision model.
    Sends the processed image and description to the user.
    
    Args:
        input_image: Path or URL to the image
        prompt: Text prompt to accompany the image
        is_local: Whether the image is a local file or URL
    """
    vision_model = (
        conf.DEFAULT_VISION_MODEL
        if is_local
        else get_setting(conf.SETTINGS_VISION_MODEL)
    )

    supports_vision = litellm.supports_vision(model=vision_model)

    if supports_vision is False:
        logger.warning(f"Unsupported vision model: {vision_model}")
        await cl.Message(
            content="",
            elements=[
                cl.Text(
                    content=f"It seems the vision model `{vision_model}` doesn't support image processing. Please choose a different model in Settings that offers Vision capabilities.",
                    display="inline",
                )
            ],
        ).send()
        return

    message = cl.Message(
        content="I'm analyzing the image. This might take a moment.",
        author=vision_model,
    )

    await message.send()
    try:
        vresponse = await litellm.acompletion(
            user=get_user_session_id(),
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": input_image}},
                    ],
                }
            ],
        )

        description = vresponse.choices[0].message.content

        if is_local:
            image = cl.Image(path=input_image, display="inline")
        else:
            image = cl.Image(url=input_image, display="inline")

        message = cl.Message(
            author=vision_model,
            content="",
            elements=[
                image,
                cl.Text(content=description, display="inline"),
            ],
            actions=[
                cl.Action(
                    name="speak_chat_response_action",
                    payload={"value": description},
                    label="Speak response",
                )
            ],
        )

        update_message_history_from_assistant(description)
        await message.send()
    except Exception as e:
        logger.error(f"Error processing image with vision model: {e}")
        await cl.Message(content=f"Failed to analyze the image: {str(e)}").send()


async def handle_trigger_async_image_gen(query: str) -> None:
    """
    Triggers asynchronous image generation using the default image generation model.
    
    Args:
        query: Text prompt for image generation
    """
    image_gen_model = conf.DEFAULT_IMAGE_GEN_MODEL
    update_message_history_from_assistant(query)

    message = cl.Message(
        content="Sure! I'll create an image based on your description. This might take a moment, please be patient.",
        author=image_gen_model,
    )
    await message.send()

    style = get_setting(conf.SETTINGS_IMAGE_GEN_IMAGE_STYLE)
    quality = get_setting(conf.SETTINGS_IMAGE_GEN_IMAGE_QUALITY)
    try:
        image_response = await litellm.aimage_generation(
            user=get_user_session_id(),
            prompt=query,
            model=image_gen_model,
            style=style,
            quality=quality,
        )

        image_gen_data = image_response["data"][0]
        image_url = image_gen_data["url"]
        revised_prompt = image_gen_data.get("revised_prompt", query)

        message = cl.Message(
            author=image_gen_model,
            content="Here's the image, along with a refined description based on your input:",
            elements=[
                cl.Image(url=image_url, display="inline"),
                cl.Text(content=revised_prompt, display="inline"),
            ],
            actions=[
                cl.Action(
                    icon="speech",
                    name="speak_chat_response_action",
                    payload={"value": revised_prompt},
                    tooltip="Speak response",
                    label="Speak response"
                )
            ],
        )

        update_message_history_from_assistant(revised_prompt)
        await message.send()

    except Exception as e:
        logger.error(f"Error generating image: {e}")
        await cl.Message(content=f"Failed to generate image: {str(e)}").send()