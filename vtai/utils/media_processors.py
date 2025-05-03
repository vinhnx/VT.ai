"""
Media processing utilities for the VT application.

Handles image, audio, and text-to-speech processing.
"""

import asyncio
import audioop
import base64
import io
import os
import tempfile
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import chainlit as cl
import litellm
import numpy as np
from openai import OpenAI
from PIL import Image

# Update imports to use vtai namespace
from vtai.utils import llm_providers_config as conf
from vtai.utils.config import get_openai_client, logger
from vtai.utils.user_session_helper import (
    get_setting,
    get_user_session_id,
    update_message_history_from_assistant,
)

# Speech-to-text settings
SILENCE_THRESHOLD = 3500  # Adjust based on your audio level (lower for quieter audio)
SILENCE_TIMEOUT = 1300.0  # Milliseconds of silence to consider the turn finished


def check_audio_capabilities(model: str) -> Tuple[bool, bool]:
    """
    Checks if a model supports audio input and output capabilities.

    Args:
        model: The model identifier to check

    Returns:
        Tuple[bool, bool]: A tuple containing (supports_audio_input, supports_audio_output)
    """
    supports_input = litellm.supports_audio_input(model=model)
    supports_output = litellm.supports_audio_output(model=model)

    logger.info(
        f"Model {model} audio capabilities - Input: {supports_input}, Output: {supports_output}"
    )

    return supports_input, supports_output


def get_audio_capable_models() -> Dict[str, Dict[str, bool]]:
    """
    Returns a dictionary of models with their audio input/output capabilities.

    Returns:
        Dict[str, Dict[str, bool]]: Dictionary mapping model names to their audio capabilities
    """
    audio_models = {}

    # Check common models
    models_to_check = [
        # OpenAI models
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4o-mini",
        # Gemini models
        "gemini/gemini-2.0-pro",
        "gemini/gemini-2.0-flash",
        # Anthropic models
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        # Default models from config
        conf.DEFAULT_AUDIO_UNDERSTANDING_MODEL,
    ]

    # Add any user-configured models from settings
    vision_model = get_setting(conf.SETTINGS_VISION_MODEL)
    if vision_model and vision_model not in models_to_check:
        models_to_check.append(vision_model)

    chat_model = get_setting(conf.SETTINGS_CHAT_MODEL)
    if chat_model and chat_model not in models_to_check:
        models_to_check.append(chat_model)

    # Check capabilities for each model
    for model in models_to_check:
        input_capable, output_capable = check_audio_capabilities(model)
        if input_capable or output_capable:
            audio_models[model] = {"input": input_capable, "output": output_capable}

    return audio_models


def get_best_audio_model(
    for_input: bool = True, for_output: bool = False
) -> Optional[str]:
    """
    Returns the best available model for audio processing based on requested capabilities.

    Args:
        for_input: Whether audio input capability is required
        for_output: Whether audio output capability is required

    Returns:
        Optional[str]: The best model that satisfies the requirements, or None if no suitable model is found
    """
    # Define priority order of models (higher quality models first)
    model_priorities = [
        "gpt-4o",
        "gemini/gemini-2.0-pro",
        "claude-3-7-sonnet-20250219",
        "gpt-4-turbo",
        "claude-3-5-sonnet-20241022",
        "gemini/gemini-2.0-flash",
        "gpt-4o-mini",
    ]

    # Get models with audio capabilities
    audio_models = get_audio_capable_models()

    # Filter models based on requirements
    suitable_models = []
    for model, capabilities in audio_models.items():
        if (not for_input or capabilities["input"]) and (
            not for_output or capabilities["output"]
        ):
            suitable_models.append(model)

    if not suitable_models:
        return None

    # Sort by priority
    for priority_model in model_priorities:
        if priority_model in suitable_models:
            return priority_model

    # If no priority model found, return the first suitable model
    return suitable_models[0]


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
        temp_filepath = os.path.join(tempfile.gettempdir(), "tts-output.mp3")

        # Using a custom timeout for the TTS request to avoid hanging connections
        with openai_client.audio.speech.with_streaming_response.create(
            model=model, voice=voice, input=context
        ) as response:
            response.stream_to_file(temp_filepath)

            # Allow a small delay for file operations to complete
            await asyncio.sleep(0.1)

            if os.path.exists(temp_filepath):
                await cl.Message(
                    author=model,
                    content="",
                    elements=[
                        cl.Audio(path=temp_filepath, display="inline"),
                        cl.Text(
                            name="TTS Info",
                            content=f"You're hearing an AI voice generated by OpenAI's {model} model, using the {voice} style. You can customize this in Settings if you'd like!",
                            display="inline",
                        ),
                    ],
                ).send()

                update_message_history_from_assistant(context)
            else:
                logger.warning("TTS file was not created successfully")

    except asyncio.CancelledError:
        logger.warning("TTS operation was cancelled")
        # Re-raise to ensure proper cleanup
        raise
    except Exception as e:
        logger.error(f"Error generating TTS response: {e}")
        await cl.Message(content=f"Failed to generate speech: {str(e)}").send()


async def handle_audio_transcribe(
    path: str, audio_file: BytesIO, openai_client: OpenAI
) -> str:
    """
    Transcribes audio to text using OpenAI's Whisper model.

    Args:
        path: Path to the audio file
        audio_file: BytesIO object for the audio file
        openai_client: OpenAI client instance

    Returns:
        The transcribed text
    """
    model = conf.DEFAULT_WHISPER_MODEL
    try:
        # Add a timeout to the transcription request
        transcription = await asyncio.wait_for(
            openai_client.audio.transcriptions.create(model=model, file=audio_file),
            timeout=30.0,  # 30 second timeout
        )
        text = transcription.text

        await cl.Message(
            content="",
            author=model,
            elements=[
                cl.Audio(path=path, display="inline"),
                cl.Text(name="Transcription", content=text, display="inline"),
            ],
        ).send()

        update_message_history_from_assistant(text)
        return text
    except asyncio.TimeoutError:
        logger.error("Audio transcription request timed out")
        await cl.Message(
            content="Audio transcription timed out. Please try with a shorter audio file."
        ).send()
        return ""
    except asyncio.CancelledError:
        logger.warning("Audio transcription was cancelled")
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        await cl.Message(content=f"Failed to transcribe audio: {str(e)}").send()
        return ""


async def handle_audio_understanding(path: str, prompt: str = None) -> None:
    """
    Analyzes and understands audio content using the best available AI model.
    Goes beyond simple transcription to provide detailed analysis of the audio content.

    Args:
        path: Path to the audio file
        prompt: Optional prompt to guide the audio understanding (default: None)
    """
    # Find the best model that supports audio input
    model = get_best_audio_model(for_input=True)

    # Fallback to the default model if no suitable model is found
    if not model:
        model = conf.DEFAULT_AUDIO_UNDERSTANDING_MODEL
        logger.warning(
            f"No models with audio input capability found. Using default model: {model}"
        )

    logger.info(f"Starting audio understanding with model: {model}")

    # Create a message to show processing status
    message = cl.Message(
        content="Processing your audio file... This may take a moment."
    )
    await message.send()

    try:
        # Get file details for logging
        file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
        logger.info(f"Audio file size: {file_size:.2f} MB")

        # Determine file extension for format
        audio_format = Path(path).suffix.lstrip(".").lower()
        logger.info(f"Audio format detected: {audio_format}")

        # Validate format
        supported_formats = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
        if audio_format not in supported_formats:
            logger.warning(
                f"Unsupported audio format: {audio_format}, defaulting to wav"
            )
            audio_format = "wav"

        # Prepare the default understanding prompt if none provided
        if not prompt:
            prompt = "What is contained in this audio recording? Please provide a detailed analysis."

        logger.info(f"Using prompt: {prompt}")

        # Get OpenAI client
        from vtai.utils.config import get_openai_client

        openai_client = get_openai_client()

        # Check if the model supports direct audio input
        supports_audio_input, _ = check_audio_capabilities(model)

        # Store the results
        analysis = ""
        transcription = ""

        if supports_audio_input:
            # Try using direct audio input if supported
            try:
                logger.info(f"Attempting direct audio processing with {model}")
                # Update message to show progress
                message.content = "Analyzing audio directly with AI..."
                await message.update()

                # Read and encode audio file
                with open(path, "rb") as audio_file:
                    encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")

                # Prepare message content for audio input
                input_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded_audio,
                                    "format": audio_format,
                                },
                            },
                        ],
                    }
                ]

                # Try with LiteLLM - ensure this is properly awaited
                completion_coroutine = litellm.acompletion(
                    user=get_user_session_id(),
                    model=model,
                    messages=input_messages,
                    modalities=["text", "audio"],
                    timeout=90.0,
                )

                response = await asyncio.wait_for(
                    completion_coroutine,
                    timeout=120.0,
                )

                # Extract analysis
                analysis = response.choices[0].message.content

                # Create full response
                full_response = f"## Audio Analysis\n\n{analysis}"

                # Update the message
                message.content = "Audio analysis complete"
                message.elements = [
                    cl.Audio(path=path, display="inline"),
                    cl.Text(
                        name="Audio Analysis", content=full_response, display="inline"
                    ),
                ]
                message.actions = [
                    cl.Action(
                        icon="speech",
                        name="speak_chat_response_action",
                        payload={"value": analysis},
                        label="Speak analysis",
                    )
                ]
                message.author = model

                update_message_history_from_assistant(full_response)
                await message.update()
                logger.info("Direct audio analysis completed successfully")
                return

            except Exception as direct_audio_error:
                # Log the error but continue to fallback approach
                logger.error(
                    f"Direct audio processing failed: {direct_audio_error}. Falling back to two-step approach."
                )

        # Fallback to two-step approach (transcribe then analyze)
        # Step 1: Transcribe with Whisper
        transcription = ""

        # Update message for transcription phase
        message.content = "Transcribing audio..."
        await message.update()

        # Transcribe with Whisper
        logger.info("Transcribing audio with Whisper")
        with open(path, "rb") as audio_file:
            audio_file_io = BytesIO(audio_file.read())
            audio_file_io.name = f"audio.{audio_format}"

            # Perform transcription - ensure this is properly awaited
            try:
                # Check if the method is already awaitable (newer OpenAI SDK)
                if hasattr(openai_client.audio.transcriptions, "acreate"):
                    # Use the explicit async method if available
                    transcription_coroutine = (
                        openai_client.audio.transcriptions.acreate(
                            model="whisper-1",
                            file=audio_file_io,
                            response_format="text",
                        )
                    )
                else:
                    # For newer versions where create() itself returns a coroutine
                    transcription_coroutine = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file_io,
                        response_format="text",
                    )

                # Ensure the coroutine is awaitable before awaiting it
                if asyncio.iscoroutine(transcription_coroutine) or hasattr(
                    transcription_coroutine, "__await__"
                ):
                    transcription_result = await asyncio.wait_for(
                        transcription_coroutine,
                        timeout=60.0,
                    )
                else:
                    # Handle non-coroutine case (synchronous result)
                    logger.warning(
                        "Transcription method returned non-awaitable, handling synchronously"
                    )
                    transcription_result = transcription_coroutine

                transcription = str(transcription_result)
                logger.info(
                    f"Transcription successful with {len(transcription)} characters"
                )
            except Exception as e:
                logger.error(f"Error in transcription: {str(e)}")
                transcription = f"[Transcription failed: {str(e)}]"

        # Log the transcription for debugging
        logger.info(f"Transcription complete: {len(transcription)} characters")

        # Step 2: Analyze the transcription
        analysis = ""

        # Update message for analysis phase
        message.content = "Analyzing transcription..."
        await message.update()

        # Create analysis prompt
        analysis_prompt = f"""
        I need you to analyze this audio transcription:

        "{transcription}"

        {prompt}

        Provide a detailed analysis of the content, tone, speakers, context, and any other relevant information.
        """

        # Use the best model for text analysis
        analysis_model = model

        # Call the API to analyze the transcription - ensure all async calls are properly awaited
        if "gemini" in model.lower():
            # Use LiteLLM for Gemini models
            gemini_completion_coroutine = litellm.acompletion(
                user=get_user_session_id(),
                model=analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that analyzes audio transcriptions.",
                    },
                    {"role": "user", "content": analysis_prompt},
                ],
                temperature=0.3,
            )

            analysis_response = await asyncio.wait_for(
                gemini_completion_coroutine,
                timeout=60.0,
            )
            analysis = analysis_response.choices[0].message.content
        else:
            # Use OpenAI directly for OpenAI models
            try:
                # Check if we should use acreate or create for async
                if hasattr(openai_client.chat.completions, "acreate"):
                    # Use explicit async method
                    openai_completion_coroutine = openai_client.chat.completions.acreate(
                        model=analysis_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that analyzes audio transcriptions.",
                            },
                            {"role": "user", "content": analysis_prompt},
                        ],
                        temperature=0.3,
                    )
                else:
                    # Standard method
                    openai_completion_coroutine = openai_client.chat.completions.create(
                        model=analysis_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that analyzes audio transcriptions.",
                            },
                            {"role": "user", "content": analysis_prompt},
                        ],
                        temperature=0.3,
                    )

                # Ensure we have an awaitable before awaiting
                if asyncio.iscoroutine(openai_completion_coroutine) or hasattr(
                    openai_completion_coroutine, "__await__"
                ):
                    analysis_response = await asyncio.wait_for(
                        openai_completion_coroutine,
                        timeout=60.0,
                    )
                else:
                    # Handle non-coroutine case
                    logger.warning(
                        "Analysis completion method returned non-awaitable, handling synchronously"
                    )
                    analysis_response = openai_completion_coroutine

                analysis = analysis_response.choices[0].message.content
                logger.info(f"Analysis successful with {len(analysis)} characters")
            except Exception as e:
                logger.error(f"Error in analysis: {str(e)}")
                analysis = f"[Analysis failed: {str(e)}]"

        # Create a combined response with both transcription and analysis
        full_response = (
            f"## Audio Transcription\n\n{transcription}\n\n## Analysis\n\n{analysis}"
        )

        # Update the message with the audio and analysis (final result)
        message.content = "Audio analysis complete"
        message.elements = [
            cl.Audio(path=path, display="inline"),
            cl.Text(name="Audio Analysis", content=full_response, display="inline"),
        ]
        message.actions = [
            cl.Action(
                icon="speech",
                name="speak_chat_response_action",
                payload={"value": analysis},
                label="Speak analysis",
            )
        ]
        message.author = model

        update_message_history_from_assistant(full_response)
        await message.update()
        logger.info("Audio analysis completed and sent to user")

    except asyncio.TimeoutError:
        logger.error("Audio processing request timed out")
        message.content = "Audio analysis timed out. The audio file might be too long or complex. Please try with a shorter recording."
        await message.update()
    except asyncio.CancelledError:
        logger.warning("Audio processing was cancelled")
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        # Provide more informative error message with troubleshooting steps
        error_message = (
            f"Failed to analyze the audio: {str(e)}\n\n"
            "Troubleshooting tips:\n"
            "1. Try a different audio file format (MP3 or WAV files work best)\n"
            "2. Ensure the audio file isn't too large (keep under 25MB)\n"
            "3. Make sure your API keys have appropriate permissions\n"
            "4. Try a shorter audio clip (under 2 minutes)"
        )
        message.content = error_message
        await message.update()


async def speech_to_text(audio_file: BinaryIO) -> str:
    """
    Transcribe speech to text using OpenAI's Whisper model.

    Args:
        audio_file: Audio file to transcribe

    Returns:
        Transcribed text from the audio
    """
    from vtai.utils.config import get_openai_client

    openai_client = get_openai_client()

    try:
        # In newer OpenAI SDK versions, create() may not be a coroutine
        logger.info("Using create for transcription (synchronous call)")
        response = openai_client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )

        # Log success and return text
        logger.info(f"Transcription successful with text: {response.text[:50]}...")
        return response.text
    except Exception as e:
        logger.error(f"Error in speech-to-text transcription: {e}", exc_info=True)
        raise e


async def process_audio() -> None:
    """
    Process the complete audio recording after silence detection.

    This function concatenates the audio chunks, creates a WAV file,
    transcribes it using Whisper, and processes the transcription.
    """
    # Get the audio buffer from the session
    audio_chunks = cl.user_session.get("audio_chunks")
    if not audio_chunks or len(audio_chunks) == 0:
        logger.warning("No audio chunks to process.")
        return

    logger.info(f"Processing {len(audio_chunks)} audio chunks")

    try:
        # Concatenate all chunks
        concatenated = np.concatenate(list(audio_chunks))
        logger.info(f"Concatenated audio length: {len(concatenated)} samples")

        # Create an in-memory binary stream
        wav_buffer = io.BytesIO()

        # Create WAV file with proper parameters
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(24000)  # sample rate (24kHz PCM)
            wav_file.writeframes(concatenated.tobytes())
            logger.info(f"Created WAV file with {wav_file.getnframes()} frames")

        # Reset buffer position
        wav_buffer.seek(0)

        # Get frames and rate info by reopening the buffer
        with wave.open(wav_buffer, "rb") as wav_info:
            frames = wav_info.getnframes()
            rate = wav_info.getframerate()
            duration = frames / float(rate)
            logger.info(f"Audio duration: {duration:.2f} seconds")

        # Reset the buffer position again for reading
        wav_buffer.seek(0)

        # Check if audio is too short
        if duration <= 1.7:
            logger.warning(
                f"The audio is too short (duration: {duration:.2f}s), discarding."
            )
            return

        audio_buffer = wav_buffer.getvalue()
        logger.info(f"Audio buffer size: {len(audio_buffer)} bytes")

        whisper_input = ("audio.wav", audio_buffer, "audio/wav")

        # Create audio element for displaying in the UI
        input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")

        # Transcribe the audio
        logger.info("Sending audio to Whisper for transcription...")
        transcription = await speech_to_text(whisper_input)
        logger.info(f"Transcription result: '{transcription}'")

        # If transcription is empty, log and return
        if not transcription or transcription.strip() == "":
            logger.warning("Received empty transcription from Whisper")
            await cl.Message(
                content="I couldn't detect any speech in your audio. Please try speaking again."
            ).send()
            return

        # Get message history
        messages = cl.user_session.get("message_history") or []

        # Send the user message with transcription
        logger.info("Sending transcription as user message")
        await cl.Message(
            author="You",
            type="user_message",
            content=transcription,
            elements=[input_audio_el],
        ).send()

        # Add transcription to message history
        messages.append({"role": "user", "content": transcription})
        cl.user_session.set("message_history", messages)

        # Get the current conversation handler based on settings
        from vtai.utils.conversation_handlers import handle_conversation

        # Create a message object for processing
        temp_message = cl.Message(content=transcription)

        # Process the transcription using the existing conversation handler
        logger.info("Handling conversation with transcription")
        await handle_conversation(temp_message, messages, None)
        logger.info("Conversation handling complete")

    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        await cl.Message(content=f"Error processing your speech: {str(e)}").send()


def encode_image_to_base64(image_path):
    """
    Encodes an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string with data URI prefix
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

            # Get image format
            img_format = Path(image_path).suffix.lstrip(".").lower()
            if img_format == "jpg":
                img_format = "jpeg"

            # Return with proper data URI format
            return f"data:image/{img_format};base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        raise e


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
                    name="Vision Model Warning",
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
        # For local images, convert to base64 if using Gemini
        image_content = input_image
        if is_local and "gemini" in vision_model.lower():
            logger.info(
                f"Converting local image to base64 for Gemini model: {vision_model}"
            )
            try:
                image_content = encode_image_to_base64(input_image)
            except Exception as e:
                logger.error(f"Failed to encode image to base64: {e}")
                await cl.Message(
                    content=f"Failed to process the image: {str(e)}"
                ).send()
                return

        # Prepare message content format for image
        input_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_content}},
                ],
            }
        ]

        # Use the Chat Completions API directly instead of trying Response API first
        logger.info(f"Using Chat Completions API for vision model: {vision_model}")
        vresponse = await asyncio.wait_for(
            litellm.acompletion(
                user=get_user_session_id(),
                model=vision_model,
                messages=input_messages,
                timeout=45.0,  # Add a specific timeout in litellm
                response_format={"type": "text"},
            ),
            timeout=60.0,  # Overall operation timeout
        )

        # With the Chat Completions API format
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
                cl.Text(name=vision_model, content=description, display="inline"),
            ],
            actions=[
                cl.Action(
                    icon="speech",
                    name="speak_chat_response_action",
                    payload={"value": description},
                    label="Speak response",
                )
            ],
        )

        update_message_history_from_assistant(description)
        await message.send()
    except asyncio.TimeoutError:
        logger.error("Vision processing request timed out")
        await cl.Message(
            content="Image analysis timed out. The image might be too complex or the service is busy."
        ).send()
    except asyncio.CancelledError:
        logger.warning("Vision processing was cancelled")
        raise
    except Exception as e:
        logger.error(f"Error processing image with vision model: {e}")
        await cl.Message(content=f"Failed to analyze the image: {str(e)}").send()


async def handle_trigger_async_image_gen(query: str) -> None:
    """
    Triggers asynchronous image generation using GPT-Image-1.

    Args:
        query: Text prompt for image generation
    """
    image_gen_model = "gpt-image-1"  # Only use GPT-Image-1
    update_message_history_from_assistant(query)

    # Create a step to show progress instead of a message
    step = cl.Step(
        name="Image Generation",
        type="generation",
        show_input=False,
    )

    async with step:
        step.input = f"Generating image using {image_gen_model}: {query}"
        await step.update()

        # Get the OpenAI client
        openai_client = get_openai_client()

        # Get GPT-Image-1 specific parameters from settings
        size = get_setting(conf.SETTINGS_IMAGE_GEN_IMAGE_SIZE) or "auto"
        quality = get_setting(conf.SETTINGS_IMAGE_GEN_IMAGE_QUALITY) or "auto"
        background = (
            get_setting(conf.SETTINGS_IMAGE_GEN_BACKGROUND)
            or conf.DEFAULT_IMAGE_GEN_BACKGROUND
        )
        output_format = (
            get_setting(conf.SETTINGS_IMAGE_GEN_OUTPUT_FORMAT)
            or conf.DEFAULT_IMAGE_GEN_OUTPUT_FORMAT
        )
        moderation = (
            get_setting(conf.SETTINGS_IMAGE_GEN_MODERATION)
            or conf.DEFAULT_IMAGE_GEN_MODERATION
        )

        # Image generation parameters for GPT-Image-1
        generation_params = {
            "model": image_gen_model,
            "prompt": query,
            "n": 1,  # Generate 1 image
        }

        # Add optional parameters only if they're not set to auto
        if size != "auto":
            generation_params["size"] = size
        if quality != "auto":
            generation_params["quality"] = quality
        if background != "auto":
            generation_params["background"] = background

        logger.info(
            f"Using GPT-Image-1 with parameters: size={size}, quality={quality}, background={background}, format={output_format}"
        )

        try:
            # Update step status
            step.output = "Generating image..."
            await step.update()

            # Generate image
            logger.info(f"Generating image with OpenAI client using GPT-Image-1")
            image_response = openai_client.images.generate(**generation_params)

            # Extract the generated image data
            image_gen_data = image_response.data[0]
            revised_prompt = getattr(image_gen_data, "revised_prompt", query)
            image_elements = []

            # GPT-Image-1 always returns b64_json
            if hasattr(image_gen_data, "b64_json"):
                logger.info("Image generated as base64 JSON. Decoding and saving...")
                try:
                    image_bytes = base64.b64decode(image_gen_data.b64_json)
                    # Save to a temporary file with appropriate extension
                    temp_image_path = os.path.join(
                        tempfile.gettempdir(),
                        f"generated_image_{int(time.time())}.{output_format}",
                    )
                    with open(temp_image_path, "wb") as f:
                        f.write(image_bytes)
                    # Allow a small delay for file operations
                    await asyncio.sleep(0.1)
                    if os.path.exists(temp_image_path):
                        # Save the image to a permanent location
                        # Create imgs directory if it doesn't exist
                        imgs_dir = os.path.join(os.getcwd(), "imgs")
                        os.makedirs(imgs_dir, exist_ok=True)

                        # Save a copy of the image to the imgs directory with a timestamp
                        img_filename = f"gpt_image_{int(time.time())}.{output_format}"
                        permanent_path = os.path.join(imgs_dir, img_filename)

                        try:
                            # Use PIL to optimize the image based on format
                            img = Image.open(temp_image_path)
                            if output_format == "png" and background == "transparent":
                                # Ensure transparency is preserved for PNG
                                if img.mode != "RGBA":
                                    img = img.convert("RGBA")
                                img.save(permanent_path, format=output_format.upper())
                                logger.info(
                                    f"Saved transparent PNG image to {permanent_path}"
                                )
                            elif output_format == "webp":
                                # WEBP with specified compression
                                compression = int(
                                    get_setting(
                                        conf.SETTINGS_IMAGE_GEN_OUTPUT_COMPRESSION
                                    )
                                    or conf.DEFAULT_IMAGE_GEN_OUTPUT_COMPRESSION
                                )
                                img.save(
                                    permanent_path,
                                    format=output_format.upper(),
                                    quality=compression,
                                )
                                logger.info(
                                    f"Saved WEBP image with {compression}% quality to {permanent_path}"
                                )
                            elif output_format == "jpeg":
                                # Convert to RGB for JPEG (no transparency)
                                if img.mode != "RGB":
                                    img = img.convert("RGB")
                                compression = int(
                                    get_setting(
                                        conf.SETTINGS_IMAGE_GEN_OUTPUT_COMPRESSION
                                    )
                                    or conf.DEFAULT_IMAGE_GEN_OUTPUT_COMPRESSION
                                )
                                img.save(
                                    permanent_path,
                                    format="JPEG",
                                    quality=compression,
                                    optimize=True,
                                )
                                logger.info(
                                    f"Saved JPEG image with {compression}% quality to {permanent_path}"
                                )
                            else:
                                # Default save
                                img.save(permanent_path, format=output_format.upper())
                                logger.info(f"Saved image to {permanent_path}")

                            # Add image to elements
                            image_elements.append(
                                cl.Image(path=permanent_path, display="inline")
                            )

                            # Create metadata text with all params
                            metadata_text = f"Model: {image_gen_model}"
                            metadata_text += f", Image format: {output_format.upper()}"
                            metadata_text += f", Quality: {quality}"
                            metadata_text += f", Size: {size}"
                            metadata_text += f", Background: {background}"

                            # Add metadata text separately with required parameters
                            if metadata_text:
                                image_elements.append(
                                    cl.Text(
                                        content=metadata_text,
                                        name="Image Info",
                                        display="inline",
                                    )
                                )
                        except Exception as img_err:
                            logger.error(
                                f"Error saving permanent image copy: {img_err}"
                            )
                            # Fallback to simple file copy if PIL processing fails
                            import shutil

                            shutil.copy2(temp_image_path, permanent_path)
                            # Still need to add the image to elements after fallback
                            image_elements.append(
                                cl.Image(path=permanent_path, display="inline")
                            )
                    else:
                        logger.error(
                            "Failed to save decoded base64 image to temporary file."
                        )
                        step.output = "Failed to process the generated image data."
                        await step.update()
                        return  # Exit if image processing failed
                except Exception as decode_err:
                    logger.error(f"Error decoding or saving base64 image: {decode_err}")
                    step.output = f"Failed to decode image data: {str(decode_err)}"
                    await step.update()
                    return  # Exit if decoding failed
            else:
                logger.error("Image generation response did not contain 'b64_json'.")
                step.output = "Received unexpected image data format from the API."
                await step.update()
                return  # Exit if format is unknown

            # Add the text element after the image element(s) if we have an image
            if image_elements:
                # Add prompt text
                if revised_prompt:
                    image_elements.append(
                        cl.Text(
                            content=revised_prompt,
                            name=f"{image_gen_model} Description",
                            display="inline",
                        )
                    )

                # Add token usage information if available
                if hasattr(image_response, "usage") and image_response.usage:
                    usage_data = image_response.usage
                    total_tokens = getattr(usage_data, "total_tokens", 0)

                    if total_tokens > 0:
                        usage_text = f"Usage: {total_tokens} total tokens"

                        # If detailed token breakdown is available
                        input_tokens = getattr(usage_data, "input_tokens", 0)
                        output_tokens = getattr(usage_data, "output_tokens", 0)

                        if input_tokens > 0 and output_tokens > 0:
                            usage_text += (
                                f" ({input_tokens} input, {output_tokens} output)"
                            )

                            # Even more detailed breakdown if available
                            input_details = getattr(
                                usage_data, "input_tokens_details", None
                            )
                            if input_details:
                                text_tokens = getattr(input_details, "text_tokens", 0)
                                image_tokens = getattr(input_details, "image_tokens", 0)
                                if text_tokens > 0 or image_tokens > 0:
                                    usage_text += f"\nInput breakdown: {text_tokens} text tokens, {image_tokens} image tokens"

                        image_elements.append(
                            cl.Text(
                                content=usage_text,
                                name="Token Usage",
                                display="inline",
                            )
                        )

                # Complete the step
                step.output = "Image generation complete!"
                await step.update()

                # Now send the final message with all elements
                await cl.Message(
                    author=image_gen_model,
                    content="Here's the image I generated based on your description:",
                    elements=image_elements,
                    actions=[
                        cl.Action(
                            icon="speech",
                            name="speak_chat_response_action",
                            payload={"value": revised_prompt},
                            tooltip="Speak description",
                            label="Speak description",
                        )
                    ],
                ).send()

                update_message_history_from_assistant(revised_prompt)
            else:
                # If we somehow got here with no image elements, send a fallback message
                step.output = "Failed to create image elements"
                await step.update()
                await cl.Message(
                    content="I generated an image but encountered an issue displaying it. Please try again."
                ).send()
        except asyncio.TimeoutError:
            logger.error("Image generation request timed out")
            step.output = "Generation timed out"
            await step.update()
            await cl.Message(
                content="Image generation timed out. Please try a simpler description or try again later."
            ).send()
        except asyncio.CancelledError:
            logger.warning("Image generation was cancelled")
            step.output = "Generation cancelled"
            await step.update()
            raise
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            step.output = f"Error: {str(e)}"
            await step.update()
            await cl.Message(content=f"Failed to generate image: {str(e)}").send()
