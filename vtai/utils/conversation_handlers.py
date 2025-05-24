"""
Conversation handler for VT application.

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
from chainlit.action import Action
from litellm.utils import trim_messages

from vtai.router.constants import SemanticRouterType
from vtai.tools.search import WebSearchOptions, WebSearchParameters, WebSearchTool
from vtai.utils import llm_providers_config as conf
from vtai.utils.config import logger
from vtai.utils.conversation_core import handle_conversation, handle_trigger_async_chat
from vtai.utils.error_handlers import safe_execution
from vtai.utils.media_processors import (
    handle_audio_transcribe,
    handle_audio_understanding,
    handle_trigger_async_image_gen,
    handle_vision,
)
from vtai.utils.settings_builder import DEFAULT_SYSTEM_PROMPT

from .supabase_client import log_request_to_supabase
from .url_extractor import extract_url
from .user_session_helper import (
    get_setting,
    get_user_email,
    get_user_id,
    get_user_session_id,
    update_message_history_from_assistant,
)


def create_message_actions(content: str, model: str) -> List[cl.Action]:
    """
    Creates standard message actions for chat responses, using Lucid icons.

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
                icon="lucide:volume-2",
                name="speak_chat_response_action",
                payload={"value": content},
                tooltip="Speak response",
                label="Speak response",
            )
        )

    # Add model change action
    actions.append(
        cl.Action(
            icon="lucide:settings",
            name="change_model_action",
            payload={"value": content},
            label=f"Using: {model}",
            description="Click to change model in settings",
        )
    )

    return actions


@cl.action_callback("change_model_action")
async def on_change_model_action(action: Action) -> None:
    """
    Handle the change model action click.
    Shows an inline text notification instructing the user how to change the model.

    Args:
        action: The action that was triggered
    """
    try:
        await action.remove()
        text_content = (
            "To change the model, click the settings gear icon in the input bar and select a "
            "different model from the 'Chat Model' dropdown."
        )
        elements = [
            cl.Text(
                name="Change Language Model", content=text_content, display="inline"
            )
        ]
        await cl.Message(content="", elements=elements).send()
    except Exception as e:
        logger.error("Error in change_model_action: %s: %s", type(e).__name__, str(e))
        await cl.Message(content="Unable to show model change instructions.").send()


async def use_chat_completion_api(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    stream_callback,
    timeout: float = 120.0,
    user_keys: dict = None,
) -> None:
    """
    Uses the Chat Completions API with enhanced Supabase logging.
    Handles streaming for the API and logs detailed usage information.

    Args:
        model: The LLM model to use
        messages: The conversation history messages
        temperature: Temperature parameter for the model
        top_p: Top-p parameter for the model
        stream_callback: Callback function to handle streaming tokens
        timeout: Timeout in seconds for the operation
        user_keys: User-specific keys for BYOK
    """
    logger.info("Using Chat Completions API for model: %s", model)

    # Get both session ID and authenticated user ID
    session_id = get_user_session_id()
    auth_user_id = get_user_id()
    user_email = get_user_email()

    # Use authenticated user ID if available, otherwise fall back to session ID
    user_for_litellm = auth_user_id if auth_user_id else session_id

    logger.info(
        "User context: session=%s, auth_user=%s, email=%s, using=%s",
        session_id[:8] + "..." if session_id else None,
        auth_user_id,
        user_email,
        user_for_litellm[:8] + "..." if user_for_litellm else None,
    )

    start_time = time.time()

    # Inject BYOK params if provided
    if user_keys:
        set_litellm_api_keys_from_settings(user_keys)

    try:
        stream = await asyncio.wait_for(
            litellm.acompletion(
                user=user_for_litellm,
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

        response_content = ""
        total_tokens = 0
        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await stream_callback(token)
                response_content += token
                total_tokens += 1  # Rough token estimation for streaming

        # Extract provider from model name
        provider = model.split("/")[0] if "/" in model else "openai"

        # Calculate cost estimate (this will be more accurate with non-streaming)
        try:
            cost_estimate = litellm.completion_cost(model=model)
        except Exception:
            cost_estimate = None

        # Enhanced logging with token tracking
        log_request_to_supabase(
            model=model,
            messages=messages,
            response={"content": response_content},
            end_user=user_for_litellm,
            status="success",
            response_time=time.time() - start_time,
            total_cost=cost_estimate,
            user_profile_id=(
                auth_user_id if auth_user_id else None
            ),  # Use authenticated user ID for profile linking
            tokens_used=total_tokens,
            provider=provider,
        )
    except Exception as e:
        logger.error("Error: %s: %s", type(e).__name__, str(e))
        # Enhanced error logging
        provider = model.split("/")[0] if "/" in model else "openai"
        log_request_to_supabase(
            model=model,
            messages=messages,
            response=None,
            end_user=user_for_litellm,
            status="failure",
            error={"error": str(e)},
            response_time=time.time() - start_time,
            user_profile_id=auth_user_id if auth_user_id else None,
            provider=provider,
        )
        # Reraise for upstream error handling
        raise


async def handle_dynamic_conversation_routing(
    messages: List[Dict[str, str]],
    model: str,
    msg: cl.Message,
    query: str,
    route_layer: Any,
    user_keys: dict = None,
) -> None:
    """
    Routes the conversation dynamically based on the semantic understanding of the user's query.

    Args:
        messages: The conversation history
        model: The LLM model to use
        msg: The chainlit message object
        query: The user's query
        route_layer: The semantic router layer
        user_keys: User-specific API keys for BYOK
    """
    async with safe_execution(operation_name="dynamic conversation routing"):
        route_choice = route_layer(query)
        route_choice_name = route_choice.name

        should_trimmed_messages = get_setting(conf.SETTINGS_TRIMMED_MESSAGES)
        if should_trimmed_messages:
            messages = trim_messages(messages, model)

        logger.info("Query: %s classified as route: %s", query, route_choice_name)

        if route_choice_name == SemanticRouterType.IMAGE_GENERATION:
            logger.info("Processing %s - Image generation", route_choice_name)
            await handle_trigger_async_image_gen(query)

        elif route_choice_name == SemanticRouterType.VISION_IMAGE_PROCESSING:
            urls = extract_url(query)
            if len(urls) > 0:
                logger.info("Processing %s - Vision with URL", route_choice_name)
                url = urls[0]
                await handle_vision(input_image=url, prompt=query, is_local=False)
            else:
                logger.info(
                    "Processing %s - No image URL, using chat", route_choice_name
                )
                await handle_trigger_async_chat(
                    llm_model=model,
                    messages=messages,
                    current_message=msg,
                    user_keys=user_keys,
                )

        elif route_choice_name == SemanticRouterType.WEB_SEARCH:
            logger.info("Processing %s - Web search", route_choice_name)
            await handle_web_search(query=query, current_message=msg)

        else:
            logger.info("Processing %s - Default chat", route_choice_name)
            await handle_trigger_async_chat(
                llm_model=model,
                messages=messages,
                current_message=msg,
                user_keys=user_keys,
            )


async def handle_files_attachment(
    message: cl.Message,
    messages: List[Dict[str, str]],
    async_openai_client: Any,
    user_keys: dict = None,
) -> None:
    """
    Handles file attachments from the user.

    Args:
        message: The user message with attachments
        messages: The conversation history
        async_openai_client: The AsyncOpenAI client
        user_keys: User-specific API keys for BYOK
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
                    await handle_conversation(
                        message, messages, None, user_keys=user_keys
                    )
                except UnicodeDecodeError:
                    # Try with a different encoding if UTF-8 fails
                    try:
                        s = p.read_text(encoding="latin-1")
                        message.content = s
                        await handle_conversation(
                            message, messages, None, user_keys=user_keys
                        )
                    except Exception as e:
                        logger.error("Error reading text file: %s", e)
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
                    logger.info("Using advanced audio understanding for file: %s", path)
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
                logger.warning("Unsupported mime type: %s", mime_type)
                await cl.Message(
                    content=f"File type {mime_type} is not supported for direct processing."
                ).send()


async def config_chat_session(settings: Dict[str, Any]) -> None:
    """
    Configures the chat session based on user settings and sets the initial system message.

    Args:
        settings: User settings dictionary
    """
    async with safe_execution(operation_name="chat session configuration"):
        # Set the model from settings
        cl.user_session.set(
            conf.SETTINGS_CHAT_MODEL, settings.get(conf.SETTINGS_CHAT_MODEL)
        )

        # Use custom system prompt if set, else default
        custom_prompt = settings.get("custom_system_prompt")
        if custom_prompt and custom_prompt.strip():
            system_prompt = custom_prompt.strip()
        else:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        system_message = {
            "role": "system",
            "content": system_prompt,
        }
        cl.user_session.set("message_history", [system_message])


async def handle_web_search(query: str, current_message: cl.Message) -> None:
    """
    Handles web search requests by using the WebSearchTool.
    Performs a web search for the query and displays the results.

    Args:
        query: The search query from the user
        current_message: The chainlit message object to stream response to
    """
    logger.info("Handling web search for query: %s", query)
    search_step = None

    try:
        # Start timing for search duration
        start = time.time()

        # Start with a web search step
        async with cl.Step(name="Web Search") as step:
            search_step = step
            # Display a searching message in the step
            await search_step.stream_token("🔍 Searching the web for: %s" % query)

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
            search_step.name = "Web Search (%ss)" % search_duration
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
            logger.error("Web search error: %s", error_message)

            # Clear the current message content and show error with fallback
            current_message.content = ""
            await current_message.stream_token(
                "I encountered an issue while searching the web: %s\n\n"
                "Let me try to answer your question about '%s' based on my knowledge instead."
                % (error_message, query)
            )

            # Fall back to regular chat completion
            model = get_setting(conf.SETTINGS_CHAT_MODEL)
            messages = cl.user_session.get("message_history") or []
            messages.append({"role": "user", "content": query})

            user_keys = cl.user_session.get("user_keys", {})
            await handle_trigger_async_chat(
                llm_model=model,
                messages=messages,
                current_message=current_message,
                user_keys=user_keys,
            )
            return

        # Clear any existing content and show search results
        current_message.content = ""

        # Add a prefix based on whether summarization was used
        if summarize_results:
            await current_message.stream_token(
                "Here's a summary of information about '%s':\n\n%s"
                % (query, response_content)
            )
        else:
            await current_message.stream_token(
                "Here's what I found on the web about '%s':\n\n%s"
                % (query, response_content)
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
                        source_text += "%s. [%s](%s)\n" % (i, title, url)

                    await current_message.stream_token(source_text)
        except Exception as e:
            logger.error("Error processing sources: %s", e)

        # Update the message content in the message history
        final_content = current_message.content
        update_message_history_from_assistant(final_content)

        # Set the actions on the message
        model_name = search_result.get("model", "WebSearch")
        current_message.actions = create_message_actions(final_content, model_name)

        # Update the message
        await current_message.update()

    except Exception as e:
        error_msg = "Error performing web search: %s" % str(e)
        logger.error(error_msg)

        # Clear any partial content and show error
        current_message.content = ""
        await current_message.stream_token(
            "I encountered an issue while searching the web: %s\n\nLet me try to answer based on my knowledge instead."
            % error_msg
        )
        await current_message.update()

        # Fall back to regular chat completion
        model = get_setting(conf.SETTINGS_CHAT_MODEL)
        messages = cl.user_session.get("message_history") or []
        messages.append({"role": "user", "content": query})

        user_keys = cl.user_session.get("user_keys", {})
        await handle_trigger_async_chat(
            llm_model=model,
            messages=messages,
            current_message=current_message,
            user_keys=user_keys,
        )


# --- SINGLE SOURCE OF TRUTH FOR LITELLM API KEY SETUP ---
def set_litellm_api_keys_from_settings(user_keys: dict) -> None:
    """
    Set the correct API key for each provider in litellm from plain text settings.
    This is the only place this logic should exist (import and use everywhere else).
    """
    if "openai" in user_keys:
        litellm.api_key = user_keys["openai"]
    if "openrouter" in user_keys:
        litellm.api_key = user_keys["openrouter"]
    if "anthropic" in user_keys:
        litellm.api_key = user_keys["anthropic"]
    if "cohere" in user_keys:
        litellm.api_key = user_keys["cohere"]
    if "mistral" in user_keys:
        litellm.api_key = user_keys["mistral"]
    if "groq" in user_keys:
        litellm.api_key = user_keys["groq"]
    if "deepseek" in user_keys:
        litellm.api_key = user_keys["deepseek"]
    if "gemini" in user_keys:
        litellm.api_key = user_keys["gemini"]
    if "gemini" in user_keys:
        litellm.api_key = user_keys["gemini"]
