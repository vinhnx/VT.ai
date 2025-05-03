"""
Conversation handling utilities for VT application.

Handles chat interactions, semantic routing, and message processing.
"""

import asyncio
import json
import os
import pathlib
import time
from typing import Any, Dict, List

import chainlit as cl
import litellm
from litellm.utils import trim_messages
from openai import AsyncOpenAI

from vtai.router.constants import SemanticRouterType
from vtai.tools.search import WebSearchOptions, WebSearchParameters, WebSearchTool
from vtai.utils import llm_providers_config as conf
from vtai.utils.config import logger
from vtai.utils.error_handlers import handle_exception, safe_execution
from vtai.utils.media_processors import (
    handle_audio_transcribe,
    handle_audio_understanding,
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


def create_message_actions(content: str, model: str) -> List[cl.Action]:
    """
    Creates standard message actions for chat responses.

    Args:
        content: The message content
        model: The model that generated the content

    Returns:
        List of actions to attach to the message
    """
    actions = []

    # Add TTS action if enabled
    enable_tts_response = get_setting(conf.SETTINGS_ENABLE_TTS_RESPONSE)
    if enable_tts_response:
        actions.append(
            cl.Action(
                icon="speech",
                name="speak_chat_response_action",
                payload={"value": content},
                tooltip="Speak response",
                label="Speak response",
            )
        )

    # Add model change action
    actions.append(
        cl.Action(
            name="change_model_action",
            payload={"value": content},
            label=f"Using: {model}",
            description="Click to change model in settings",
        )
    )

    return actions


async def use_chat_completion_api(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    stream_callback,
    timeout: float = 120.0,
) -> None:
    """
    Uses the Chat Completions API directly instead of trying Responses API first.
    Handles streaming for the API.

    Args:
        model: The LLM model to use
        messages: The conversation history messages
        temperature: Temperature parameter for the model
        top_p: Top-p parameter for the model
        stream_callback: Callback function to handle streaming tokens
        timeout: Timeout in seconds for the operation
    """
    logger.info(f"Using Chat Completions API for model: {model}")
    stream = await asyncio.wait_for(
        litellm.acompletion(
            user=get_user_session_id(),
            model=model,
            messages=messages,
            stream=True,
            num_retries=3,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout
            - 30.0,  # Set LiteLLM timeout slightly less than our overall timeout
            response_format={"type": "text"},
        ),
        timeout=timeout,
    )

    # Process Chat Completions API format
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await stream_callback(token)


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

    async def on_timeout():
        await current_message.stream_token(
            "\n\nI apologize, but the response timed out. Please try again with a shorter query."
        )
        await current_message.update()

    async with safe_execution(
        operation_name=f"chat completion with model {llm_model}", on_timeout=on_timeout
    ):
        # Use the helper function for API calls
        await use_chat_completion_api(
            model=llm_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            stream_callback=current_message.stream_token,
        )

        # After successful streaming, update the message content and history
        content = current_message.content
        update_message_history_from_assistant(content)

        # Set the actions on the message using the helper function
        current_message.actions = create_message_actions(content, llm_model)

        await current_message.update()


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
    # Use "assistant" as the author name to match the avatar file in /public/avatars/
    msg = cl.Message(content="")
    await msg.send()

    query = message.content
    update_message_history_from_user(query)

    async with safe_execution(operation_name="conversation handling"):
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
    async with safe_execution(operation_name="dynamic conversation routing"):
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

        elif route_choice_name == SemanticRouterType.WEB_SEARCH:
            logger.info(f"Processing {route_choice_name} - Web search")
            await handle_web_search(query=query, current_message=msg)

        else:
            logger.info(f"Processing {route_choice_name} - Default chat")
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

    async with safe_execution(operation_name="file attachment handling"):
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
                # Check if a prompt was provided for audio understanding or just use transcription
                if prompt and (
                    "analyze" in prompt.lower()
                    or "understand" in prompt.lower()
                    or "explain" in prompt.lower()
                ):
                    logger.info(f"Using advanced audio understanding for file: {path}")
                    # Use the new audio understanding with the user's prompt
                    await handle_audio_understanding(path, prompt)
                else:
                    # For simple transcription or if no specific prompt, use the existing transcription
                    f = pathlib.Path(path)
                    await handle_audio_transcribe(path, f, async_openai_client)

                    # If transcription completed but user might want more analysis, suggest it
                    if not prompt:
                        await cl.Message(
                            content="I've transcribed your audio. If you'd like a more detailed analysis of its content, upload it again with a prompt like 'analyze this audio' or 'explain what's in this recording'."
                        ).send()

            else:
                logger.warning(f"Unsupported mime type: {mime_type}")
                await cl.Message(
                    content=f"File type {mime_type} is not supported for direct processing."
                ).send()


async def config_chat_session(settings: Dict[str, Any]) -> None:
    """
    Configures the chat session based on user settings and sets the initial system message.

    Args:
        settings: User settings dictionary
    """
    from vtai.assistants.mino.mino import INSTRUCTIONS
    from vtai.utils.chat_profile import AppChatProfileType

    async with safe_execution(operation_name="chat session configuration"):
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

    # Create empty objects that will be initialized in the context
    thinking_step = None
    final_answer = None
    thinking = False

    async with safe_execution(
        operation_name=f"thinking conversation with model {model}"
    ):
        # Start with a thinking step
        async with cl.Step(name="Thinking") as step:
            thinking_step = step
            # Create a message for the final answer, but don't send it yet
            final_answer = cl.Message(content="")

            # Define a custom streaming callback that handles thinking tags
            async def thinking_stream_callback(content):
                nonlocal thinking, thinking_step, final_answer

                # Skip if content is None or empty
                if not content:
                    return

                # Check for exact <think> tag
                if content == "<think>":
                    thinking = True
                    return

                # Check for exact </think> tag
                elif content == "</think>":
                    thinking = False
                    # Update the thinking step with duration
                    thought_for = round(time.time() - start)
                    thinking_step.name = f"Thought for {thought_for}s"
                    await thinking_step.update()
                    return

                # Stream content based on thinking flag
                if thinking:
                    # Stream to thinking step
                    await thinking_step.stream_token(content)
                else:
                    # Stream to final answer
                    await final_answer.stream_token(content)

            # Use the API fallback helper
            await use_chat_completion_api(
                model=model,
                messages=thinking_messages,
                temperature=float(temperature),
                top_p=float(top_p),
                stream_callback=thinking_stream_callback,
                timeout=120.0,
            )

        # Send the final answer after thinking is complete
        if not final_answer.content:
            # If no final answer was provided, create a fallback message
            await cl.Message(
                content="I've thought about this but don't have a specific answer to provide."
            ).send()
        else:
            # Update message history and add TTS action
            content = final_answer.content
            update_message_history_from_assistant(content)

            # Set the actions on the message using the helper function
            final_answer.actions = create_message_actions(content, model)

            await final_answer.send()


async def handle_deepseek_reasoner_conversation(
    message: cl.Message, messages: List[Dict[str, str]], route_layer: Any
) -> None:
    """
    Handles conversations using DeepSeek Reasoner model with its native reasoning capability.
    Shows the AI's reasoning process before presenting the final answer.
    Uses DeepSeek's reasoning_content attribute for the thinking process.

    Args:
        message: The user message object
        messages: The conversation history
        route_layer: The semantic router layer
    """
    # Track start time for the thinking duration
    start = time.time()

    # Get settings
    deepseek_api_key = os.getenv("DEEP_SEEK_API_KEY")
    if not deepseek_api_key:
        await cl.Message(
            content="DeepSeek API key not found. Please set the DEEP_SEEK_API_KEY environment variable."
        ).send()
        return

    model = "deepseek-reasoner"  # DeepSeek Reasoner model
    query = message.content.strip()
    update_message_history_from_user(query)

    # Create DeepSeek client
    client = AsyncOpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    # Variables that will be initialized in the context
    thinking_step = None
    final_answer = None
    thinking_completed = False

    async with safe_execution(
        operation_name=f"deepseek reasoner conversation with model {model}"
    ):
        # Prepare messages for the API call
        deepseek_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
        ]

        # Add conversation history
        for msg in messages:
            if msg["role"] != "system":  # Skip system messages from history
                deepseek_messages.append(msg)

        # Add the current user query
        deepseek_messages.append({"role": "user", "content": query})

        # Create the stream
        stream = await client.chat.completions.create(
            model=model,
            messages=deepseek_messages,
            stream=True,
        )

        # Start with a thinking step
        async with cl.Step(name="Thinking") as step:
            thinking_step = step
            # Create a message for the final answer, but don't send it yet
            final_answer = cl.Message(content="")

            # Process the reasoning content first
            async for chunk in stream:
                delta = chunk.choices[0].delta
                reasoning_content = getattr(delta, "reasoning_content", None)

                if reasoning_content is not None and not thinking_completed:
                    # Stream the reasoning content to the thinking step
                    await thinking_step.stream_token(reasoning_content)
                elif not thinking_completed:
                    # Exit the thinking step when reasoning is complete
                    thought_for = round(time.time() - start)
                    thinking_step.name = f"Thought for {thought_for}s"
                    await thinking_step.update()
                    thinking_completed = True
                    break

        # Stream the final answer content
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                await final_answer.stream_token(delta.content)

        # Send the final answer after thinking is complete
        if not final_answer.content:
            # If no final answer was provided, create a fallback message
            await cl.Message(
                content="I've thought about this but don't have a specific answer to provide.",
            ).send()
        else:
            # Update message history and add TTS action
            content = final_answer.content
            update_message_history_from_assistant(content)

            # Set the actions on the message using the helper function
            final_answer.actions = create_message_actions(content, model)

            await final_answer.send()


async def handle_web_search(query: str, current_message: cl.Message) -> None:
    """
    Handles web search requests by using the WebSearchTool.
    Performs a web search for the query and displays the results.

    Args:
        query: The search query from the user
        current_message: The chainlit message object to stream response to
    """
    logger.info(f"Handling web search for query: {query}")
    search_step = None

    try:
        # Start timing for search duration
        start = time.time()

        # Start with a web search step
        async with cl.Step(name="Web Search") as step:
            search_step = step
            # Display a searching message in the step
            await search_step.stream_token(f"🔍 Searching the web for: {query}")

            # Get OpenAI API key from environment
            openai_api_key = os.environ.get("OPENAI_API_KEY")

            # Get Tavily API key from environment if available
            tavily_api_key = os.environ.get("TAVILY_API_KEY")

            # Check if summarization setting exists in user settings
            summarize_setting = get_setting("SETTINGS_SUMMARIZE_SEARCH_RESULTS")
            # Default to True if the setting doesn't exist
            summarize_results = True if summarize_setting is None else summarize_setting

            use_tavily = tavily_api_key is not None

            # Initialize the web search tool with both API keys
            web_search_tool = WebSearchTool(
                api_key=openai_api_key, tavily_api_key=tavily_api_key
            )

            # Create search parameters
            params = WebSearchParameters(
                query=query,
                model="openai/gpt-4o",
                max_results=5,
                search_options=WebSearchOptions(
                    search_context_size="medium",
                    include_urls=True,
                    summarize_results=summarize_results,
                ),
                use_tavily=use_tavily,
            )

            # Log which search method we're using
            if use_tavily:
                logger.info("Using Tavily for web search")
                await search_step.stream_token("\nUsing Tavily search engine...")
            else:
                logger.info("Using LiteLLM function calling for web search")
                await search_step.stream_token(
                    "\nUsing OpenAI function calling for search..."
                )

            if summarize_results:
                await search_step.stream_token(
                    "\nSummarization enabled - synthesizing results..."
                )

            # Add some visual indicator that search is in progress
            await asyncio.sleep(0.5)
            await search_step.stream_token(".")
            await asyncio.sleep(0.5)
            await search_step.stream_token(".")

            # Perform the search
            search_result = await web_search_tool.search(params)

            # Calculate search duration
            search_duration = round(time.time() - start)
            search_step.name = f"Web Search ({search_duration}s)"
            await search_step.update()

            # Small delay to let the user see the completed step
            await asyncio.sleep(1)

        # Get the response content
        response_content = search_result.get("response", "No search results available")

        # Get search status
        search_status = search_result.get("status", "unknown")

        # Check if search had an error
        if search_status == "error":
            error_message = search_result.get("error", "Unknown error occurred")
            logger.error(f"Web search error: {error_message}")

            # Clear the current message content and show error with fallback
            current_message.content = ""
            await current_message.stream_token(
                f"I encountered an issue while searching the web: {error_message}\n\n"
                f"Let me try to answer your question about '{query}' based on my knowledge instead."
            )

            # Fall back to regular chat completion
            model = get_setting(conf.SETTINGS_CHAT_MODEL)
            messages = cl.user_session.get("message_history") or []
            messages.append({"role": "user", "content": query})

            await handle_trigger_async_chat(
                llm_model=model, messages=messages, current_message=current_message
            )
            return

        # Clear any existing content and show search results
        current_message.content = ""

        # Add a prefix based on whether summarization was used
        if summarize_results:
            await current_message.stream_token(
                f"Here's a summary of information about '{query}':\n\n{response_content}"
            )
        else:
            await current_message.stream_token(
                f"Here's what I found on the web about '{query}':\n\n{response_content}"
            )

        # Try to extract and display sources if available
        try:
            sources_json = search_result.get("sources_json")
            if sources_json:
                sources = json.loads(sources_json).get("sources", [])
                if sources:
                    source_text = "\n\n**Sources:**\n"
                    for i, source in enumerate(sources, 1):
                        title = source.get("title", "Untitled")
                        url = source.get("url", "No URL")
                        source_text += f"{i}. [{title}]({url})\n"

                    await current_message.stream_token(source_text)
        except Exception as e:
            logger.error(f"Error processing sources: {e}")

        # Update the message content in the message history
        final_content = current_message.content
        update_message_history_from_assistant(final_content)

        # Set the actions on the message
        model_name = search_result.get("model", "WebSearch")
        current_message.actions = create_message_actions(final_content, model_name)

        # Update the message
        await current_message.update()

    except Exception as e:
        error_msg = f"Error performing web search: {str(e)}"
        logger.error(error_msg)

        # Clear any partial content and show error
        current_message.content = ""
        await current_message.stream_token(
            f"I encountered an issue while searching the web: {error_msg}\n\nLet me try to answer based on my knowledge instead."
        )
        await current_message.update()

        # Fall back to regular chat completion
        model = get_setting(conf.SETTINGS_CHAT_MODEL)
        messages = cl.user_session.get("message_history") or []
        messages.append({"role": "user", "content": query})

        await handle_trigger_async_chat(
            llm_model=model, messages=messages, current_message=current_message
        )
