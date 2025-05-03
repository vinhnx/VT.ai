"""
VT - Main application entry point.

A multimodal AI chat application with dynamic conversation routing.
"""

import argparse
import asyncio
import audioop
import json
import os
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import chainlit as cl
import dotenv
import numpy as np
from chainlit.types import ChatProfile

# Import modules
from vtai.assistants.manager import get_or_create_assistant
from vtai.utils import constants as const
from vtai.utils import llm_providers_config as conf
from vtai.utils.assistant_tools import process_thread_message, process_tool_call
from vtai.utils.config import initialize_app, logger
from vtai.utils.conversation_handlers import (
    config_chat_session,
    handle_conversation,
    handle_files_attachment,
    handle_thinking_conversation,
)
from vtai.utils.dict_to_object import DictToObject
from vtai.utils.error_handlers import handle_exception
from vtai.utils.file_handlers import process_files
from vtai.utils.llm_profile_builder import build_llm_profile
from vtai.utils.media_processors import handle_tts_response
from vtai.utils.safe_execution import safe_execution
from vtai.utils.settings_builder import build_settings
from vtai.utils.user_session_helper import get_setting, is_in_assistant_profile

# Initialize the application with improved client configuration
route_layer, assistant_id, openai_client, async_openai_client = initialize_app()
# Removed the debugging code for printing python executable and pip list

# Initialize or retrieve the assistant (with web search capabilities)
import os


async def init_assistant():
    """Initialize or retrieve the OpenAI assistant with web search enabled."""
    global assistant_id  # Moved to the beginning of the function

    # Print all environment variables
    print("All environment variables:", os.environ)

    try:
        # Get or create the assistant with web search capabilities
        assistant = await get_or_create_assistant(
            client=async_openai_client,
            assistant_id=assistant_id,
            name=const.APP_NAME,
            instructions="You are a helpful assistant with web search capabilities. When information might be outdated or not in your training data, you can search the web for more current information.",
            model="gpt-4o",
        )

        # Store the assistant ID globally
        assistant_id = assistant.id

        logger.info(f"Successfully initialized assistant: {assistant.id}")
        return assistant.id
    except Exception as e:
        logger.error(f"Error initializing assistant: {e}")
        # Return the existing assistant_id if any
        return assistant_id


# App name constant
APP_NAME = const.APP_NAME


@cl.set_chat_profiles
async def build_chat_profile(_=None):
    """Define and set available chat profiles."""
    # Force shuffling of starters on each app startup
    # This ensures starter prompts are in a different order each time
    return [
        ChatProfile(
            name=profile.title,
            markdown_description=profile.description,
            starters=conf.get_shuffled_starters(use_random=True),
        )
        for profile in conf.APP_CHAT_PROFILES
    ]


@cl.on_chat_start
async def start_chat():
    """
    Initialize the chat session with settings and system message.
    """
    # Initialize default settings
    cl.user_session.set(conf.SETTINGS_CHAT_MODEL, conf.DEFAULT_MODEL)

    # Build LLM profile with direct icon path instead of using map
    build_llm_profile()

    # Settings configuration
    settings = await build_settings()

    # Configure chat session with selected model
    await config_chat_session(settings)

    if is_in_assistant_profile():
        try:
            # Initialize or get the assistant with web search capabilities
            assistant_id_result = await init_assistant()
            if not assistant_id_result:
                raise ValueError("Failed to initialize assistant")

            # Create a new thread for the conversation
            thread = await async_openai_client.beta.threads.create()
            cl.user_session.set("thread", thread)
            logger.info("Created new thread: %s", thread.id)
        except (
            asyncio.TimeoutError,
            ConnectionError,
            ValueError,
        ) as e:
            logger.error("Failed to create thread: %s", e)
            await handle_exception(e)
        except Exception as e:
            logger.error("Unexpected error creating thread: %s", repr(e))
            await handle_exception(e)


@asynccontextmanager
async def managed_run_execution(thread_id, run_id):
    """
    Context manager to safely handle run execution and ensure proper cleanup.

    Args:
        thread_id: Thread ID
        run_id: Run ID
    """
    try:
        yield
    except asyncio.CancelledError:
        logger.warning("Run execution canceled for run %s", run_id)
        try:
            # Attempt to cancel the run if it was cancelled externally
            await async_openai_client.beta.threads.runs.cancel(
                thread_id=thread_id, run_id=run_id
            )
        except Exception as e:
            logger.error("Error cancelling run: %s", e)
        raise
    except Exception as e:
        logger.error("Error in run execution: %s", e)
        await handle_exception(e)


async def process_code_interpreter_tool(
    step_references: Dict[str, cl.Step], step: Any, tool_call: Any
) -> Dict[str, Any]:
    """
    Process code interpreter tool calls.

    Args:
        step_references: Dictionary of step references
        step: The run step
        tool_call: The tool call to process

    Returns:
        Tool output dictionary
    """
    output_value = ""
    if (
        tool_call.code_interpreter.outputs
        and len(tool_call.code_interpreter.outputs) > 0
    ):
        output_value = tool_call.code_interpreter.outputs[0]

    # Create a step for code execution
    async with cl.Step(
        name="Code Interpreter",
        type="code",
        parent_id=(
            cl.context.current_step.id
            if hasattr(cl.context, "current_step") and cl.context.current_step
            else None
        ),
    ) as code_step:
        code_step.input = tool_call.code_interpreter.input or "# Generating code"

        # Stream tokens to show activity
        await code_step.stream_token("Executing code")
        await asyncio.sleep(0.3)  # Small delay for visibility
        await code_step.stream_token(".")
        await asyncio.sleep(0.3)
        await code_step.stream_token(".")

        # Update with output when available
        if output_value:
            code_step.output = output_value
            await code_step.update()

    await process_tool_call(
        step_references=step_references,
        step=step,
        tool_call=tool_call,
        name=tool_call.type,
        input=tool_call.code_interpreter.input or "# Generating code",
        output=output_value,
        show_input="python",
    )

    return {
        "output": tool_call.code_interpreter.outputs or "",
        "tool_call_id": tool_call.id,
    }


async def process_function_tool(
    step_references: Dict[str, cl.Step], step: Any, tool_call: Any
) -> Dict[str, Any]:
    """
    Process function tool calls.

    Args:
        step_references: Dictionary of step references
        step: The run step
        tool_call: The tool call to process

    Returns:
        Tool output dictionary
    """
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    # Handle the web search tool specifically
    if function_name == "web_search":
        from vtai.tools.search import (
            WebSearchOptions,
            WebSearchParameters,
            WebSearchTool,
        )

        # Get API keys from environment
        openai_api_key = os.environ.get("OPENAI_API_KEY") or None
        tavily_api_key = os.environ.get("TAVILY_API_KEY") or None

        # Determine if we should use Tavily
        use_tavily = tavily_api_key is not None
        if use_tavily:
            logger.info("Using Tavily for assistant web search")

        # Initialize the web search tool with appropriate API keys
        web_search_tool = WebSearchTool(
            api_key=openai_api_key, tavily_api_key=tavily_api_key
        )

        # Extract search parameters
        query = function_args.get("query", "")
        model = function_args.get("model", "openai/gpt-4o")
        max_results = function_args.get("max_results", None)

        # Build search options if provided
        search_options = None
        if any(
            key in function_args
            for key in ["search_context_size", "include_urls", "summarize_results"]
        ):
            search_options = WebSearchOptions(
                search_context_size=function_args.get("search_context_size", "medium"),
                include_urls=function_args.get("include_urls", True),
                summarize_results=function_args.get("summarize_results", True),
            )
        else:
            # Default search options
            search_options = WebSearchOptions(
                search_context_size="medium", include_urls=True, summarize_results=True
            )

        # Create search parameters
        params = WebSearchParameters(
            query=query,
            model=model,
            max_results=max_results,
            search_options=search_options,
            use_tavily=use_tavily,
        )

        # Perform the search
        try:
            logger.info(f"Performing web search for: {query}")

            # Create a step for the web search execution
            async with cl.Step(
                name=f"Web Search: {query}",
                type="tool",
                parent_id=(
                    cl.context.current_step.id
                    if hasattr(cl.context, "current_step") and cl.context.current_step
                    else None
                ),
            ) as search_step:
                search_step.input = f"Searching for: {query}"

                # Stream tokens to show activity
                await search_step.stream_token("Searching")
                await asyncio.sleep(0.5)  # Small delay for visibility
                await search_step.stream_token(".")
                await asyncio.sleep(0.5)
                await search_step.stream_token(".")

                # Indicate if we're summarizing
                if search_options.summarize_results:
                    await search_step.stream_token("\nSummarization enabled...")

                # Execute the search
                search_result = await web_search_tool.search(params)

                # Get search status
                search_status = search_result.get("status", "unknown")

                # Check if search had an error
                if search_status == "error":
                    error_msg = search_result.get("error", "Unknown error occurred")
                    logger.error(f"Web search error: {error_msg}")

                    # Update step with error info
                    search_step.output = f"Error performing web search: {error_msg}"
                    await search_step.update()

                    await process_tool_call(
                        step_references=step_references,
                        step=step,
                        tool_call=tool_call,
                        name=function_name,
                        input=function_args,
                        output=f"Error performing web search: {error_msg}",
                        show_input="json",
                    )

                    return {
                        "output": f"Error performing web search: {error_msg}",
                        "tool_call_id": tool_call.id,
                    }

                # Process the results
                response_content = search_result.get(
                    "response", "No search results available"
                )

                # Add source information if available
                sources_text = ""
                try:
                    sources_json = search_result.get("sources_json")
                    if sources_json:
                        sources = json.loads(sources_json).get("sources", [])
                        if sources:
                            sources_text = "\n\nSources:\n"
                            for i, source in enumerate(sources, 1):
                                title = source.get("title", "Untitled")
                                url = source.get("url", "No URL")
                                sources_text += f"{i}. {title} - {url}\n"

                    # Append sources to response if available
                    if sources_text:
                        response_content = f"{response_content}\n{sources_text}"
                except Exception as e:
                    logger.error(f"Error processing sources: {e}")

                # Update step with final result
                search_step.output = f"Found information about '{query}'"
                await search_step.update()

            await process_tool_call(
                step_references=step_references,
                step=step,
                tool_call=tool_call,
                name=function_name,
                input=function_args,
                output=response_content,
                show_input="json",
            )

            return {
                "output": response_content,
                "tool_call_id": tool_call.id,
            }
        except Exception as e:
            error_msg = f"Error performing web search: {str(e)}"
            logger.error(error_msg)

            await process_tool_call(
                step_references=step_references,
                step=step,
                tool_call=tool_call,
                name=function_name,
                input=function_args,
                output=error_msg,
                show_input="json",
            )

            return {
                "output": error_msg,
                "tool_call_id": tool_call.id,
            }
    # For other tools that are temporarily disabled
    logger.warning(
        "Function tool call received but tools are disabled: %s", function_name
    )

    await process_tool_call(
        step_references=step_references,
        step=step,
        tool_call=tool_call,
        name=function_name,
        input=function_args,
        output="Function tools are temporarily disabled",
        show_input="json",
    )

    return {
        "output": "Function tools are temporarily disabled",
        "tool_call_id": tool_call.id,
    }


async def process_retrieval_tool(
    step_references: Dict[str, cl.Step], step: Any, tool_call: Any
) -> None:
    """
    Process retrieval tool calls.

    Args:
        step_references: Dictionary of step references
        step: The run step
        tool_call: The tool call to process
    """
    # Create a step for retrieval execution
    async with cl.Step(
        name="Document Retrieval",
        type="retrieval",
        parent_id=(
            cl.context.current_step.id
            if hasattr(cl.context, "current_step") and cl.context.current_step
            else None
        ),
    ) as retrieval_step:
        retrieval_step.input = "Retrieving relevant information from uploaded documents"

        # Stream tokens to show activity
        await retrieval_step.stream_token("Retrieving")
        await asyncio.sleep(0.3)  # Small delay for visibility
        await retrieval_step.stream_token(".")
        await asyncio.sleep(0.3)
        await retrieval_step.stream_token(".")

        # Update with completion indication
        retrieval_step.output = "Retrieved relevant information from documents"
        await retrieval_step.update()

    await process_tool_call(
        step_references=step_references,
        step=step,
        tool_call=tool_call,
        name=tool_call.type,
        input="Retrieving information",
        output="Retrieved information",
    )


async def create_run_instance(thread_id: str) -> Any:
    """
    Create a run instance for the assistant.

    Args:
        thread_id: Thread ID to create run for

    Returns:
        Run instance object
    """
    global assistant_id

    # Ensure we have a valid assistant ID
    if not assistant_id:
        # Try to initialize the assistant if not already done
        assistant_id_result = await init_assistant()
        if not assistant_id_result:
            logger.error("Could not create or retrieve assistant")
            raise ValueError(
                "No assistant ID available. Please configure an assistant ID"
            )
        assistant_id = assistant_id_result

    # Create a run with the assistant
    logger.info(f"Creating run for thread {thread_id} with assistant {assistant_id}")
    return await async_openai_client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )


async def process_tool_calls(
    step_details: Any, step_references: Dict[str, cl.Step], step: Any
) -> List[Dict[str, Any]]:
    """
    Process all tool calls from a step.

    Args:
        step_details: The step details object
        step_references: Dictionary of step references
        step: The run step

    Returns:
        List of tool outputs
    """
    tool_outputs = []

    if step_details.type != "tool_calls":
        return tool_outputs

    for tool_call in step_details.tool_calls:
        if isinstance(tool_call, dict):
            tool_call = DictToObject(tool_call)

        if tool_call.type == "code_interpreter":
            output = await process_code_interpreter_tool(
                step_references, step, tool_call
            )
            tool_outputs.append(output)
        elif tool_call.type == "retrieval":
            await process_retrieval_tool(step_references, step, tool_call)
        elif tool_call.type == "function":
            output = await process_function_tool(step_references, step, tool_call)
            tool_outputs.append(output)

    return tool_outputs


@cl.step(name=APP_NAME, type="run")
async def run(thread_id: str, human_query: str, file_ids: Optional[List[str]] = None):
    """
    Run the assistant with the user query and manage the response.

    Args:
        thread_id: Thread ID to interact with
        human_query: User's message
        file_ids: Optional list of file IDs to attach
    """
    # Add the message to the thread
    file_ids = file_ids or []
    try:
        # Add message to thread with timeout
        await asyncio.wait_for(
            async_openai_client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=human_query,
            ),
            timeout=30.0,
        )

        # Create the run
        run_instance = await create_run_instance(thread_id)

        message_references: Dict[str, cl.Message] = {}
        step_references: Dict[str, cl.Step] = {}
        tool_outputs = []

        # Use context manager for safer execution
        async with managed_run_execution(thread_id, run_instance.id):
            # Periodically check for updates with a timeout for each operation
            while True:
                run_instance = await asyncio.wait_for(
                    async_openai_client.beta.threads.runs.retrieve(
                        thread_id=thread_id, run_id=run_instance.id
                    ),
                    timeout=30.0,
                )

                # Fetch the run steps with timeout
                run_steps = await asyncio.wait_for(
                    async_openai_client.beta.threads.runs.steps.list(
                        thread_id=thread_id, run_id=run_instance.id, order="asc"
                    ),
                    timeout=30.0,
                )

                for step in run_steps.data:
                    # Fetch step details with timeout
                    run_step = await asyncio.wait_for(
                        async_openai_client.beta.threads.runs.steps.retrieve(
                            thread_id=thread_id, run_id=run_instance.id, step_id=step.id
                        ),
                        timeout=30.0,
                    )
                    step_details = run_step.step_details

                    # Process message creation
                    if step_details.type == "message_creation":
                        thread_message = await asyncio.wait_for(
                            async_openai_client.beta.threads.messages.retrieve(
                                message_id=step_details.message_creation.message_id,
                                thread_id=thread_id,
                            ),
                            timeout=30.0,
                        )
                        await process_thread_message(
                            message_references, thread_message, async_openai_client
                        )

                    # Process tool calls
                    tool_outputs.extend(
                        await process_tool_calls(step_details, step_references, step)
                    )

                # Submit tool outputs if required
                if (
                    run_instance.status == "requires_action"
                    and hasattr(run_instance, "required_action")
                    and run_instance.required_action is not None
                    and hasattr(run_instance.required_action, "type")
                    and run_instance.required_action.type == "submit_tool_outputs"
                    and tool_outputs
                ):
                    await asyncio.wait_for(
                        async_openai_client.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread_id,
                            run_id=run_instance.id,
                            tool_outputs=tool_outputs,
                        ),
                        timeout=30.0,
                    )

                # Wait between polling to reduce API load
                await asyncio.sleep(2)

                if run_instance.status in [
                    "cancelled",
                    "failed",
                    "completed",
                    "expired",
                ]:
                    logger.info(
                        "Run %s finished with status: %s",
                        run_instance.id,
                        run_instance.status,
                    )
                    break

    except asyncio.TimeoutError:
        logger.error("Timeout occurred during run execution")
        await cl.Message(
            content="The operation timed out. Please try again with a simpler query."
        ).send()
    except Exception as e:
        logger.error("Error in run: %s", e)
        await handle_exception(e)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Handle incoming user messages and route them appropriately.

    Args:
        message: The user message object
    """
    async with safe_execution(
        operation_name="message processing",
        cancelled_message="The operation was cancelled. Please try again.",
    ):
        if is_in_assistant_profile():
            thread: Thread = cl.user_session.get("thread")
            files_ids = await process_files(message.elements, async_openai_client)
            await run(
                thread_id=thread.id, human_query=message.content, file_ids=files_ids
            )
        else:
            # Get message history
            messages = cl.user_session.get("message_history") or []

            # Check if current model is a reasoning model that benefits from <think>
            current_model = get_setting(conf.SETTINGS_CHAT_MODEL)
            is_reasoning = conf.is_reasoning_model(current_model)

            # If this is a reasoning model and <think> is not already in content, add it
            if is_reasoning and "<think>" not in message.content:
                # Clone the original message content
                original_content = message.content
                # Modify the message content to include <think> tag
                message.content = f"<think>{original_content}"
                logger.info(
                    "Automatically added <think> tag for reasoning model: %s",
                    current_model,
                )

            if message.elements and len(message.elements) > 0:
                await handle_files_attachment(message, messages, async_openai_client)
            else:
                # Check for <think> tag directly in user request
                if "<think>" in message.content.lower():
                    logger.info(
                        "Processing message with <think> tag using thinking "
                        "conversation handler"
                    )
                    await handle_thinking_conversation(message, messages, route_layer)
                else:
                    await handle_conversation(message, messages, route_layer)


@cl.on_settings_update
async def update_settings(settings: Dict[str, Any]) -> None:
    """
    Update user settings based on preferences.

    Args:
        settings: Dictionary of user settings
    """
    try:
        # Update temperature if provided
        if settings_temperature := settings.get(conf.SETTINGS_TEMPERATURE):
            cl.user_session.set(conf.SETTINGS_TEMPERATURE, settings_temperature)

        # Update top_p if provided
        if settings_top_p := settings.get(conf.SETTINGS_TOP_P):
            cl.user_session.set(conf.SETTINGS_TOP_P, settings_top_p)

        # Check if chat model was changed
        model_changed = False
        if conf.SETTINGS_CHAT_MODEL in settings:
            cl.user_session.set(
                conf.SETTINGS_CHAT_MODEL, settings.get(conf.SETTINGS_CHAT_MODEL)
            )
            model_changed = True

        # Update all other settings
        setting_keys = [
            conf.SETTINGS_IMAGE_GEN_IMAGE_STYLE,
            conf.SETTINGS_IMAGE_GEN_IMAGE_QUALITY,
            conf.SETTINGS_VISION_MODEL,
            conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING,
            conf.SETTINGS_TTS_MODEL,
            conf.SETTINGS_TTS_VOICE_PRESET_MODEL,
            conf.SETTINGS_ENABLE_TTS_RESPONSE,
            conf.SETTINGS_TRIMMED_MESSAGES,
        ]

        for key in setting_keys:
            if key in settings:
                cl.user_session.set(key, settings.get(key))

        # If model was changed, rebuild the chat profiles to ensure icons are properly set
        if model_changed:
            # Rebuild LLM profiles to ensure icons are updated
            build_llm_profile()
            logger.info("Chat model changed, rebuilt profiles with icons")

        logger.info("Settings updated successfully")
    except Exception as e:
        logger.error("Error updating settings: %s", e)


@cl.action_callback("speak_chat_response_action")
async def on_speak_chat_response(action: cl.Action) -> None:
    """
    Handle TTS action triggered by the user.

    Args:
        action: The action object containing payload
    """
    try:
        await action.remove()
        value = action.payload.get("value") or ""
        await handle_tts_response(value, openai_client)
    except Exception as e:
        logger.error("Error handling TTS response: %s", e)
        await cl.Message(content="Failed to generate speech. Please try again.").send()


@cl.on_audio_start
async def on_audio_start() -> bool:
    """
    Initialize audio recording session when the user starts speaking.

    This function sets up the necessary state variables for tracking speech
    and silence during audio recording.

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

    This function analyzes incoming audio chunks, measures silence duration,
    and triggers processing when the user stops speaking.

    Args:
        chunk: Audio chunk from the user
    """
    from vtai.utils.media_processors import (
        SILENCE_THRESHOLD,
        SILENCE_TIMEOUT,
        process_audio,
    )

    # Get audio chunks from user session
    audio_chunks = cl.user_session.get("audio_chunks")

    # If this is the first chunk, initialize timers and state
    if chunk.isStart:
        logger.info("Starting new audio recording session")
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        cl.user_session.set("silent_duration_ms", 0)

        # Ensure audio_chunks is initialized
        if audio_chunks is None:
            audio_chunks = []
            cl.user_session.set("audio_chunks", audio_chunks)
        return

    # Safety check - ensure audio_chunks exists
    if audio_chunks is None:
        logger.warning("audio_chunks is None, reinitializing")
        audio_chunks = []
        cl.user_session.set("audio_chunks", audio_chunks)

    # Process the audio chunk
    audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
    audio_chunks.append(audio_chunk)

    # Get session variables with safety defaults
    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms", 0)
    is_speaking = cl.user_session.get("is_speaking", True)

    # Safety check for last_elapsed_time
    if last_elapsed_time is None:
        logger.warning("last_elapsed_time is None, resetting to current time")
        last_elapsed_time = chunk.elapsedTime
        cl.user_session.set("last_elapsed_time", last_elapsed_time)
        # Skip time diff calculation for this iteration
        time_diff_ms = 0
    else:
        # Calculate the time difference between this chunk and the previous one
        time_diff_ms = chunk.elapsedTime - last_elapsed_time

    # Update the last elapsed time for the next iteration
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # Compute the RMS (root mean square) energy of the audio chunk
    audio_energy = audioop.rms(
        chunk.data, 2
    )  # Assumes 16-bit audio (2 bytes per sample)

    logger.debug(
        f"Audio energy: {audio_energy}, Silent duration: {silent_duration_ms}ms"
    )

    if audio_energy < SILENCE_THRESHOLD:
        # Audio is considered silent
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        if silent_duration_ms >= SILENCE_TIMEOUT and is_speaking:
            logger.info(
                f"Silence detected for {silent_duration_ms}ms, processing audio"
            )
            cl.user_session.set("is_speaking", False)
            await process_audio()
    else:
        # Audio is not silent, reset silence timer and mark as speaking
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            logger.info("Speech resumed after silence")
            cl.user_session.set("is_speaking", True)


def copy_if_newer(src: Path, dst: Path, log_msg: str = None) -> bool:
    """
    Copy a file only if the source is newer than the destination or if destination doesn't exist.

    Args:
        src: Source file path
        dst: Destination file path
        log_msg: Optional message to log if copy occurs

    Returns:
        bool: True if file was copied, False otherwise
    """
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)
        if log_msg:
            logger.info(f"{log_msg} to {dst}")
        return True
    return False


def create_symlink_or_empty_file(src: Path, dst: Path, is_dir: bool = False) -> None:
    """
    Creates a symlink if possible, otherwise creates an empty directory or file.

    Args:
        src: Source path to link to
        dst: Destination path for the symlink
        is_dir: Whether the source is a directory
    """
    if dst.exists():
        if dst.is_symlink():
            # Already a symlink, no action needed
            return
        elif is_dir:
            # It's a regular directory, rename it as backup
            backup_dir = dst.parent / f"{dst.name}.backup"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            dst.rename(backup_dir)

    # Create the symlink
    os.symlink(str(src), str(dst))
    logger.info(f"Created symlink from {dst} to {src}")


def setup_chainlit_config():
    """
    Sets up a centralized Chainlit configuration directory and creates necessary files.

    Returns:
        Path: Path to the centralized chainlit config directory
    """
    # Get the package installation directory
    pkg_dir = Path(__file__).parent.parent
    src_chainlit_dir = pkg_dir / ".chainlit"

    # Create centralized Chainlit config directory
    user_config_dir = Path(os.path.expanduser("~/.config/vtai"))
    chainlit_config_dir = user_config_dir / ".chainlit"

    # Create directories
    user_config_dir.mkdir(parents=True, exist_ok=True)
    chainlit_config_dir.mkdir(parents=True, exist_ok=True)

    # Handle configuration files
    if src_chainlit_dir.exists() and src_chainlit_dir.is_dir():
        # Copy config.toml if needed
        copy_if_newer(
            src=src_chainlit_dir / "config.toml",
            dst=chainlit_config_dir / "config.toml",
            log_msg="Copied default config.toml",
        )

        # Handle translations directory
        src_translations = src_chainlit_dir / "translations"
        dst_translations = chainlit_config_dir / "translations"

        if src_translations.exists() and src_translations.is_dir():
            dst_translations.mkdir(exist_ok=True)

            # Copy translation files
            for trans_file in src_translations.glob("*.json"):
                copy_if_newer(
                    src=trans_file,
                    dst=dst_translations / trans_file.name,
                    log_msg=None,  # Don't log individual translation files
                )

            logger.info(f"Copied translations to {dst_translations}")

    # Handle chainlit.md
    src_md = pkg_dir / "chainlit.md"
    central_md = user_config_dir / "chainlit.md"

    if src_md.exists() and src_md.stat().st_size > 0:
        copy_if_newer(src=src_md, dst=central_md, log_msg="Copied custom chainlit.md")
    elif not central_md.exists():
        # Create empty file
        central_md.touch()
        logger.info(f"Created empty chainlit.md at {central_md}")

    # Create symlinks if not in project directory
    current_dir = Path.cwd()
    local_chainlit_dir = current_dir / ".chainlit"
    local_md = current_dir / "chainlit.md"

    if str(current_dir) != str(pkg_dir.parent):
        try:
            # Handle .chainlit directory symlink
            create_symlink_or_empty_file(
                src=chainlit_config_dir, dst=local_chainlit_dir, is_dir=True
            )

            # Handle chainlit.md file - create empty file to prevent Chainlit defaults
            if not local_md.exists():
                local_md.touch()
                logger.info(f"Created empty chainlit.md to prevent default content")

        except Exception as e:
            # Fallback if symlink creation fails
            logger.warning(f"Could not create symlinks: {e}. Using local files.")
            local_chainlit_dir.mkdir(exist_ok=True)

            if not local_md.exists():
                try:
                    local_md.touch()
                    logger.info(f"Created empty chainlit.md as fallback")
                except Exception as create_error:
                    logger.warning(
                        f"Failed to create empty chainlit.md: {create_error}"
                    )

    return chainlit_config_dir


def main():
    """
    Entry point for the VT.ai application when installed via pip.
    This function is called when the 'vtai' command is executed.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="VT.ai Application")
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (e.g., deepseek, sonnet, o3-mini)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key in the format provider=key (e.g., openai=sk-..., anthropic=sk-...)",
    )

    # Parse known args to handle chainlit's own arguments
    args, remaining_args = parser.parse_known_args()

    # Create user config directory
    config_dir = Path(os.path.expanduser("~/.config/vtai"))
    config_dir.mkdir(parents=True, exist_ok=True)

    # Set Chainlit's config path before any imports, using environment variables
    # that Chainlit recognizes for its paths
    os.environ["CHAINLIT_CONFIG_DIR"] = str(config_dir)
    os.environ["CHAINLIT_HOME"] = str(config_dir)

    # Set up centralized Chainlit configuration
    chainlit_config_dir = setup_chainlit_config()

    env_path = config_dir / ".env"

    # Process API key if provided
    if args.api_key:
        try:
            # Parse provider=key format
            if "=" in args.api_key:
                provider, key = args.api_key.split("=", 1)

                # Map provider to appropriate environment variable name
                provider_map = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "deepseek": "DEEPSEEK_API_KEY",
                    "cohere": "COHERE_API_KEY",
                    "huggingface": "HUGGINGFACE_API_KEY",
                    "groq": "GROQ_API_KEY",
                    "openrouter": "OPENROUTER_API_KEY",
                    "gemini": "GEMINI_API_KEY",
                    "mistral": "MISTRAL_API_KEY",
                    "tavily": "TAVILY_API_KEY",
                    "lmstudio": "LM_STUDIO_API_KEY",
                }

                env_var = provider_map.get(provider.lower())
                if env_var:
                    # Create or update .env file with the API key
                    dotenv.set_key(env_path, env_var, key)
                    print(f"API key for {provider} saved to {env_path}")
                else:
                    print(
                        f"Unknown provider: {provider}. Supported providers are: {', '.join(provider_map.keys())}"
                    )
                    return
            else:
                print("API key format should be provider=key (e.g., openai=sk-...)")
                return
        except Exception as e:
            print(f"Error saving API key: {e}")
            return

    # Directly load the .env file we just created/updated
    dotenv.load_dotenv(env_path)

    # Initialize command to run
    cmd_args = []

    # Add model selection if specified
    if args.model:
        # Use the model parameter to set the appropriate environment variable
        model_map = {
            "deepseek": (
                "DEEPSEEK_API_KEY",
                "You need to provide a DeepSeek API key with --api-key deepseek=<key>",
            ),
            "sonnet": (
                "ANTHROPIC_API_KEY",
                "You need to provide an Anthropic API key with --api-key anthropic=<key>",
            ),
            "o3-mini": (
                "OPENAI_API_KEY",
                "You need to provide an OpenAI API key with --api-key openai=<key>",
            ),
            # Add more model mappings here
        }

        if args.model.lower() in model_map:
            env_var, error_msg = model_map[args.model.lower()]
            if not os.getenv(env_var):
                print(error_msg)
                return

            # Set model in environment for the chainlit process
            os.environ["VT_DEFAULT_MODEL"] = args.model
            print(f"Using model: {args.model}")
        else:
            print(
                f"Unknown model: {args.model}. Supported models are: {', '.join(model_map.keys())}"
            )
            return

    # Check for the chainlit run command in remaining args
    if not remaining_args or "run" not in remaining_args:
        # No run command provided, directly run the app using chainlit
        cmd = f"chainlit run {os.path.realpath(__file__)}"
    else:
        # Pass any arguments to chainlit
        cmd = f"chainlit {' '.join(remaining_args)} {os.path.realpath(__file__)}"

    print(f"Starting VT.ai: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    main()
