"""
Voice conversation example for VT.ai using Chainlit.

This example demonstrates a speech-to-text interface using:
- OpenAI's Whisper for speech-to-text
- OpenAI's GPT models for conversation

The app enables real-time voice input with silence detection
to automatically determine when the user has finished speaking.
"""

import audioop
import io
import os
import wave
from typing import Any, BinaryIO, Dict, List, Optional

import chainlit as cl
import numpy as np
from openai import AsyncOpenAI

# Environment variable for API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Validate required environment variable
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set")

# Define a threshold for detecting silence and a timeout for ending a turn
SILENCE_THRESHOLD = 3500  # Adjust based on your audio level (lower for quieter audio)
SILENCE_TIMEOUT = 1300.0  # Seconds of silence to consider the turn finished


@cl.step(type="tool")
async def speech_to_text(audio_file: BinaryIO) -> str:
    """
    Transcribe speech to text using OpenAI's Whisper model.

    Args:
        audio_file: Audio file to transcribe

    Returns:
        Transcribed text from the audio
    """
    response = await openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


@cl.step(type="tool")
async def generate_text_answer(transcription: str) -> str:
    """
    Generate a text response to the transcribed speech using OpenAI.

    Args:
        transcription: Transcribed text from speech

    Returns:
        Generated response text
    """
    message_history = cl.user_session.get("message_history")

    message_history.append({"role": "user", "content": transcription})

    response = await openai_client.chat.completions.create(
        model="gpt-4o", messages=message_history, temperature=0.2
    )

    message = response.choices[0].message
    message_history.append({"role": "assistant", "content": message.content})

    return message.content


@cl.on_chat_start
async def start() -> None:
    """
    Initialize the chat session when a user starts a conversation.
    """
    cl.user_session.set("message_history", [])
    await cl.Message(
        content="Welcome to Speech-to-Text demo! Press `p` to talk!",
    ).send()


@cl.on_audio_start
async def on_audio_start() -> bool:
    """
    Initialize audio recording session when the user starts speaking.

    Returns:
        True to allow audio recording
    """
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk) -> None:
    """
    Process each audio chunk as it arrives, detecting silence for turn-taking.

    Args:
        chunk: Audio chunk from the user
    """
    audio_chunks = cl.user_session.get("audio_chunks")

    if audio_chunks is not None:
        audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)

    # If this is the first chunk, initialize timers and state
    if chunk.isStart:
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        return

    audio_chunks = cl.user_session.get("audio_chunks")
    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")

    # Calculate the time difference between this chunk and the previous one
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # Compute the RMS (root mean square) energy of the audio chunk
    audio_energy = audioop.rms(
        chunk.data, 2
    )  # Assumes 16-bit audio (2 bytes per sample)

    if audio_energy < SILENCE_THRESHOLD:
        # Audio is considered silent
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        if silent_duration_ms >= SILENCE_TIMEOUT and is_speaking:
            cl.user_session.set("is_speaking", False)
            await process_audio()
    else:
        # Audio is not silent, reset silence timer and mark as speaking
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            cl.user_session.set("is_speaking", True)


async def process_audio() -> None:
    """
    Process the complete audio recording after silence detection.

    This function concatenates the audio chunks, creates a WAV file,
    transcribes it using Whisper, and generates a text response.
    """
    # Get the audio buffer from the session
    audio_chunks = cl.user_session.get("audio_chunks")
    if not audio_chunks or len(audio_chunks) == 0:
        print("No audio chunks to process.")
        return

    # Concatenate all chunks
    concatenated = np.concatenate(list(audio_chunks))

    # Create an in-memory binary stream
    wav_buffer = io.BytesIO()

    # Create WAV file with proper parameters
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(24000)  # sample rate (24kHz PCM)
        wav_file.writeframes(concatenated.tobytes())

    # Reset buffer position
    wav_buffer.seek(0)

    # Get frames and rate info by reopening the buffer
    with wave.open(wav_buffer, "rb") as wav_info:
        frames = wav_info.getnframes()
        rate = wav_info.getframerate()
        duration = frames / float(rate)

    # Reset the buffer position again for reading
    wav_buffer.seek(0)

    # Check if audio is too short
    if duration <= 1.7:
        print("The audio is too short, please try again.")
        return

    audio_buffer = wav_buffer.getvalue()

    input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")

    whisper_input = ("audio.wav", audio_buffer, "audio/wav")
    transcription = await speech_to_text(whisper_input)

    await cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        elements=[input_audio_el],
    ).send()

    answer = await generate_text_answer(transcription)

    await cl.Message(content=answer).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Handle text messages from the user.

    Args:
        message: User message object
    """
    await cl.Message(content="This is a speech-to-text demo, press P to start!").send()


if __name__ == "__main__":
    # This allows running the demo directly
    import sys

    # Install dependencies if needed
    try:
        import audioop

        import numpy
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "uv", "pip", "install", "numpy"])

    # Run the Chainlit app
    import os

    os.system(f"chainlit run {__file__}")
