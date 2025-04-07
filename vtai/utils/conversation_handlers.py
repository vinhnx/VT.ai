"""
Conversation handling utilities for VT.ai application.

Handles chat interactions, semantic routing, and message processing.
"""

import asyncio
import pathlib
import time
from typing import Any, Dict, List

import chainlit as cl
import litellm
from litellm.utils import trim_messages

from vtai.router.constants import SemanticRouterType
from vtai.utils import llm_provider_settings as conf
from vtai.utils.config import logger
from vtai.utils.error_handlers import handle_exception
from vtai.utils.media_processors import (
    handle_audio_transcribe,
    handle_trigger_async_image_gen,
    handle_vision,
)
from vtai.utils.url_extractor import extract_url
from vtai.utils.user_session_helper import (
    get_setting,
    get_user_session_id,
    update_message_history_from_assistant,
    update_message_history_from_user,
)


async def handle_trigger_async_chat(
    llm_model: str, messages: List[Dict[str, str]], current_message: cl.Message
) -> None:
    """
    Triggers an asynchronous chat completion using the specified LLM model.
    Streams the response back to the user and updates the message history.

    Args:
        llm_model: The LLM model to use
        messages: The conversation history messages
        current_message: The chainlit message object to stream response to
    """
    temperature = get_setting(conf.SETTINGS_TEMPERATURE)
    top_p = get_setting(conf.SETTINGS_TOP_P)

    try:
        # Set a reasonable timeout for completion
        completion_timeout = 120.0  # 2 minutes

        # Create the stream with timeout handling
        stream = await asyncio.wait_for(
            litellm.acompletion(
                model=llm_model,
                messages=messages,
                stream=True,
                num_retries=3,
                temperature=temperature,
                top_p=top_p,
                timeout=90.0,  # Set LiteLLM timeout slightly less than our overall timeout
            ),
            timeout=completion_timeout,
        )

        # Process the stream safely with proper cancellation handling
        try:
            async for part in stream:
                if token := part.choices[0].delta.content or "":
                    await current_message.stream_token(token)
        except asyncio.CancelledError:
            logger.warning("Stream processing was cancelled")
            # Make sure we handle cancellation properly
            if hasattr(stream, "aclose") and callable(stream.aclose):
                await stream.aclose()
            elif hasattr(stream, "aclose_async") and callable(stream.aclose_async):
                await stream.aclose_async()
            raise

        # After successful streaming, update the message content and history
        content = current_message.content
        update_message_history_from_assistant(content)

        # Add TTS action if enabled
        enable_tts_response = get_setting(conf.SETTINGS_ENABLE_TTS_RESPONSE)
        if enable_tts_response:
            current_message.actions = [
                cl.Action(
                    icon="speech",
                    name="speak_chat_response_action",
                    payload={"value": content},
                    tooltip="Speak response",
                    label="Speak response",
                )
            ]

        await current_message.update()

    except asyncio.TimeoutError:
        logger.error(f"Timeout while processing chat completion with model {llm_model}")
        await current_message.stream_token(
            "\n\nI apologize, but the response timed out. Please try again with a shorter query."
        )
        await current_message.update()
    except asyncio.CancelledError:
        logger.warning("Chat completion was cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in handle_trigger_async_chat: {e}")
        await handle_exception(e)


async def handle_conversation(
    message: cl.Message, messages: List[Dict[str, str]], route_layer: Any
) -> None:
    """
    Handles text-based conversations with the user.
    Routes the conversation based on settings and semantic understanding.

    Args:
        message: The user message object
        messages: The conversation history
        route_layer: The semantic router layer
    """
    model = get_setting(conf.SETTINGS_CHAT_MODEL)
    msg = cl.Message(content="", author=model)
    await msg.send()

    query = message.content
    update_message_history_from_user(query)

    try:
        use_dynamic_conversation_routing = get_setting(
            conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING
        )

        if use_dynamic_conversation_routing and route_layer:
            await handle_dynamic_conversation_routing(
                messages, model, msg, query, route_layer
            )
        else:
            await handle_trigger_async_chat(
                llm_model=model, messages=messages, current_message=msg
            )
    except asyncio.CancelledError:
        logger.warning("Conversation handling was cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in handle_conversation: {e}")
        await handle_exception(e)


async def handle_dynamic_conversation_routing(
    messages: List[Dict[str, str]],
    model: str,
    msg: cl.Message,
    query: str,
    route_layer: Any,
) -> None:
    """
    Routes the conversation dynamically based on the semantic understanding of the user's query.

    Args:
        messages: The conversation history
        model: The LLM model to use
        msg: The chainlit message object
        query: The user's query
        route_layer: The semantic router layer
    """
    try:
        route_choice = route_layer(query)
        route_choice_name = route_choice.name

        should_trimmed_messages = get_setting(conf.SETTINGS_TRIMMED_MESSAGES)
        if should_trimmed_messages:
            messages = trim_messages(messages, model)

        logger.info(f"Query: {query} classified as route: {route_choice_name}")

        if route_choice_name == SemanticRouterType.IMAGE_GENERATION:
            logger.info(f"Processing {route_choice_name} - Image generation")
            await handle_trigger_async_image_gen(query)

        elif route_choice_name == SemanticRouterType.VISION_IMAGE_PROCESSING:
            urls = extract_url(query)
            if len(urls) > 0:
                logger.info(f"Processing {route_choice_name} - Vision with URL")
                url = urls[0]
                await handle_vision(input_image=url, prompt=query, is_local=False)
            else:
                logger.info(
                    f"Processing {route_choice_name} - No image URL, using chat"
                )
                await handle_trigger_async_chat(
                    llm_model=model, messages=messages, current_message=msg
                )
        else:
            logger.info(f"Processing {route_choice_name} - Default chat")
            await handle_trigger_async_chat(
                llm_model=model, messages=messages, current_message=msg
            )
    except asyncio.CancelledError:
        logger.warning("Dynamic conversation routing was cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in handle_dynamic_conversation_routing: {e}")
        await handle_exception(e)
        # Fallback to regular chat if routing fails
        await handle_trigger_async_chat(
            llm_model=model, messages=messages, current_message=msg
        )


async def handle_files_attachment(
    message: cl.Message, messages: List[Dict[str, str]], async_openai_client: Any
) -> None:
    """
    Handles file attachments from the user.

    Args:
        message: The user message with attachments
        messages: The conversation history
        async_openai_client: The AsyncOpenAI client
    """
    if not message.elements:
        await cl.Message(content="No file attached").send()
        return

    prompt = message.content

    try:
        for file in message.elements:
            path = str(file.path)
            mime_type = file.mime or ""

            if "image" in mime_type:
                await handle_vision(path, prompt=prompt, is_local=True)

            elif "text" in mime_type:
                try:
                    p = pathlib.Path(path)
                    s = p.read_text(encoding="utf-8")
                    message.content = s
                    await handle_conversation(message, messages, None)
                except UnicodeDecodeError:
                    # Try with a different encoding if UTF-8 fails
                    try:
                        s = p.read_text(encoding="latin-1")
                        message.content = s
                        await handle_conversation(message, messages, None)
                    except Exception as e:
                        logger.error(f"Error reading text file: {e}")
                        await cl.Message(
                            content=f"Failed to read text file: {str(e)}"
                        ).send()

            elif "audio" in mime_type:
                f = pathlib.Path(path)
                await handle_audio_transcribe(path, f, async_openai_client)

            else:
                logger.warning(f"Unsupported mime type: {mime_type}")
                await cl.Message(
                    content=f"File type {mime_type} is not supported for direct processing."
                ).send()

    except asyncio.CancelledError:
        logger.warning("File attachment handling was cancelled")
        raise
    except Exception as e:
        logger.error(f"Error handling file attachment: {e}")
        await handle_exception(e)


async def config_chat_session(settings: Dict[str, Any]) -> None:
    """
    Configures the chat session based on user settings and sets the initial system message.

    Args:
        settings: User settings dictionary
    """
    from vtai.assistants.mino.mino import INSTRUCTIONS
    from vtai.utils.chat_profile import AppChatProfileType

    try:
        chat_profile = cl.user_session.get("chat_profile")
        if chat_profile == AppChatProfileType.CHAT.value:
            cl.user_session.set(
                conf.SETTINGS_CHAT_MODEL, settings.get(conf.SETTINGS_CHAT_MODEL)
            )

            system_message = {
                "role": "system",
                "content": "You are a helpful assistant who tries their best to answer questions: ",
            }

            cl.user_session.set("message_history", [system_message])

        elif chat_profile == AppChatProfileType.ASSISTANT.value:
            system_message = {"role": "system", "content": INSTRUCTIONS}

            cl.user_session.set("message_history", [system_message])

            msg = "Hello! I'm Mino, your Assistant. I'm here to assist you. Please don't hesitate to ask me anything you'd like to know. Currently, I can write and run code to answer math questions."
            await cl.Message(content=msg).send()
    except Exception as e:
        logger.error(f"Error configuring chat session: {e}")
        await handle_exception(e)


async def handle_thinking_conversation(
    message: cl.Message, messages: List[Dict[str, str]], route_layer: Any
) -> None:
    """
    Handles conversations with visible thinking process.
    Shows the AI's reasoning before presenting the final answer.
    Uses <think> and </think> tags within the model's response to toggle between thinking and final answer.

    Args:
        message: The user message object
        messages: The conversation history
        route_layer: The semantic router layer
    """
    # Track start time for the thinking duration
    start = time.time()

    # Get model and settings
    model = get_setting(conf.SETTINGS_CHAT_MODEL)
    temperature = get_setting(conf.SETTINGS_TEMPERATURE) or 0.7
    top_p = get_setting(conf.SETTINGS_TOP_P) or 0.9

    # Remove the <think> tag from the query if it exists in the user message
    query = message.content.replace("<think>", "").strip()
    update_message_history_from_user(query)

    # Add special instruction to the model to use <think> tags
    thinking_messages = [m.copy() for m in messages]
    # Add a system message with instructions about using <think> tags
    thinking_messages.append(
        {
            "role": "system",
            "content": "When responding to this query, first use <think> tag to show your reasoning process, then close it with </think> and provide your final answer.",
        }
    )
    # Add the user query
    thinking_messages.append({"role": "user", "content": query})

    try:
        # Create the stream
        stream = await litellm.acompletion(
            user=get_user_session_id(),
            model=model,
            messages=thinking_messages,
            temperature=float(temperature),
            top_p=float(top_p),
            stream=True,
        )

        # Flag to track if we're currently in thinking mode
        thinking = False

        # Start with a thinking step
        async with cl.Step(name="Thinking") as thinking_step:
            # Create a message for the final answer, but don't send it yet
            final_answer = cl.Message(content="", author=model)

            # Process the stream
            async for chunk in stream:
                delta = chunk.choices[0].delta
                content = delta.content or ""

                # Check for <think> tag
                if "<think>" in content:
                    thinking = True
                    # Remove the tag from content
                    content = content.replace("<think>", "")

                # Check for </think> tag
                if "</think>" in content:
                    thinking = False
                    # Update the thinking step with duration
                    thought_for = round(time.time() - start)
                    thinking_step.name = f"Thought for {thought_for}s"
                    await thinking_step.update()
                    # Remove the tag from content
                    content = content.replace("</think>", "")

                if content:
                    if thinking:
                        # Stream to thinking step
                        await thinking_step.stream_token(content)
                    else:
                        # Stream to final answer
                        await final_answer.stream_token(content)

        # Send the final answer after thinking is complete
        if not final_answer.content:
            # If no final answer was provided, create a fallback message
            await cl.Message(
                content="I've thought about this but don't have a specific answer to provide.",
                author=model,
            ).send()
        else:
            # Update message history and add TTS action
            content = final_answer.content
            update_message_history_from_assistant(content)

            # Add TTS action if enabled
            enable_tts_response = get_setting(conf.SETTINGS_ENABLE_TTS_RESPONSE)
            if enable_tts_response:
                final_answer.actions = [
                    cl.Action(
                        icon="speech",
                        name="speak_chat_response_action",
                        payload={"value": content},
                        tooltip="Speak response",
                        label="Speak response",
                    )
                ]

            await final_answer.send()

    except asyncio.TimeoutError:
        logger.error(f"Timeout while processing chat completion with model {model}")
        await cl.Message(
            content="\n\nI apologize, but the response timed out. Please try again with a shorter query."
        ).send()
    except asyncio.CancelledError:
        logger.warning("Chat completion was cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in handle_thinking_conversation: {e}")
        await handle_exception(e)
