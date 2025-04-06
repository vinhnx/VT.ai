"""
Conversation handling utilities for VT.ai application.

Handles chat interactions, semantic routing, and message processing.
"""

import pathlib
from typing import Any, Dict, List, Optional, Tuple

import chainlit as cl
import litellm
from litellm.utils import trim_messages
from openai.types.beta.thread import Thread

from router.constants import SemanticRouterType
from utils import llm_settings_config as conf
from utils.config import logger
from utils.error_handlers import handle_exception
from utils.media_processors import (
    handle_audio_transcribe,
    handle_trigger_async_image_gen,
    handle_vision,
)
from utils.url_extractor import extract_url
from utils.user_session_helper import (
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
        # use LiteLLM for other providers
        stream = await litellm.acompletion(
            model=llm_model,
            messages=messages,
            stream=True,
            num_retries=2,
            temperature=temperature,
            top_p=top_p,
        )

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await current_message.stream_token(token)

        content = current_message.content
        update_message_history_from_assistant(content)

        enable_tts_response = get_setting(conf.SETTINGS_ENABLE_TTS_RESPONSE)
        if enable_tts_response:
            current_message.actions = [
                cl.Action(
                    icon="speech",
                    name="speak_chat_response_action",
                    payload={"value": content},
                    tooltip="Speak response",
                    label="Speak response"
                )
            ]

        await current_message.update()

    except Exception as e:
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

    use_dynamic_conversation_routing = get_setting(
        conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING
    )

    if use_dynamic_conversation_routing:
        await handle_dynamic_conversation_routing(
            messages, model, msg, query, route_layer
        )
    else:
        await handle_trigger_async_chat(
            llm_model=model, messages=messages, current_message=msg
        )


async def handle_dynamic_conversation_routing(
    messages: List[Dict[str, str]], 
    model: str, 
    msg: cl.Message, 
    query: str,
    route_layer: Any
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
            logger.info(f"Processing {route_choice_name} - No image URL, using chat")
            await handle_trigger_async_chat(
                llm_model=model, messages=messages, current_message=msg
            )
    else:
        logger.info(f"Processing {route_choice_name} - Default chat")
        await handle_trigger_async_chat(
            llm_model=model, messages=messages, current_message=msg
        )


async def handle_files_attachment(
    message: cl.Message, 
    messages: List[Dict[str, str]],
    async_openai_client: Any
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

    for file in message.elements:
        path = str(file.path)
        mime_type = file.mime or ""

        if "image" in mime_type:
            await handle_vision(path, prompt=prompt, is_local=True)

        elif "text" in mime_type:
            p = pathlib.Path(path)
            s = p.read_text(encoding="utf-8")
            message.content = s
            await handle_conversation(message, messages, None)  # Pass None for route_layer as it's not used for text files

        elif "audio" in mime_type:
            f = pathlib.Path(path)
            await handle_audio_transcribe(path, f, async_openai_client)
        
        else:
            logger.warning(f"Unsupported mime type: {mime_type}")
            await cl.Message(content=f"File type {mime_type} is not supported for direct processing.").send()


async def config_chat_session(settings: Dict[str, Any]) -> None:
    """
    Configures the chat session based on user settings and sets the initial system message.
    
    Args:
        settings: User settings dictionary
    """
    from assistants.mino.mino import INSTRUCTIONS
    from utils.chat_profile import AppChatProfileType

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

        msg = "Hello! I'm here to assist you. Please don't hesitate to ask me anything you'd like to know."
        await cl.Message(content=msg).send()

    elif chat_profile == AppChatProfileType.ASSISTANT.value:
        system_message = {"role": "system", "content": INSTRUCTIONS}

        cl.user_session.set("message_history", [system_message])

        msg = "Hello! I'm Mino, your Assistant. I'm here to assist you. Please don't hesitate to ask me anything you'd like to know. Currently, I can write and run code to answer math questions."
        await cl.Message(content=msg).send()