"""
VT.ai - Main application entry point.

A multimodal AI chat application with dynamic conversation routing.
"""

import json
import asyncio
import time
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import chainlit as cl
import litellm
from openai import AsyncOpenAI, OpenAI
from openai.types.beta.thread import Thread

import utils.constants as const
from assistants.mino.create_assistant import tool_map
from assistants.mino.mino import MinoAssistant
from utils import llm_settings_config as conf
from utils.assistant_tools import process_thread_message, process_tool_call
from utils.config import initialize_app, logger
from utils.conversation_handlers import (
    config_chat_session,
    handle_conversation,
    handle_files_attachment,
    handle_thinking_conversation,
)
from utils.error_handlers import handle_exception
from utils.file_handlers import process_files
from utils.llm_profile_builder import build_llm_profile
from utils.media_processors import handle_tts_response
from utils.settings_builder import build_settings
from utils.user_session_helper import (
    is_in_assistant_profile,
    get_setting,
    get_user_session_id,
    update_message_history_from_assistant,
    update_message_history_from_user
)
from utils.dict_to_object import DictToObject

# Initialize the application with improved client configuration
route_layer, assistant_id, openai_client, async_openai_client = initialize_app()

# App name constant
APP_NAME = const.APP_NAME


@cl.set_chat_profiles
async def build_chat_profile(user=None):
    """Define and set available chat profiles."""
    return conf.CHAT_PROFILES


@cl.on_chat_start
async def start_chat():
    """
    Initialize the chat session with settings and system message.
    """
    # Initialize default settings
    cl.user_session.set(
        conf.SETTINGS_CHAT_MODEL, "default_model_name"
    )

    # Build LLM profile
    build_llm_profile(conf.ICONS_PROVIDER_MAP)

    # Settings configuration
    settings = await build_settings()

    # Configure chat session with selected model
    await config_chat_session(settings)

    if is_in_assistant_profile():
        try:
            thread = await async_openai_client.beta.threads.create()
            cl.user_session.set("thread", thread)
            logger.info(f"Created new thread: {thread.id}")
        except Exception as e:
            logger.error(f"Failed to create thread: {e}")
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
        logger.warning(f"Run execution canceled for run {run_id}")
        try:
            # Attempt to cancel the run if it was cancelled externally
            await async_openai_client.beta.threads.runs.cancel(
                thread_id=thread_id,
                run_id=run_id
            )
        except Exception as e:
            logger.error(f"Error cancelling run: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in run execution: {e}")
        await handle_exception(e)


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
        init_message = await asyncio.wait_for(
            async_openai_client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=human_query,
            ),
            timeout=30.0
        )

        # Create the run
        if not assistant_id:
            mino = MinoAssistant(openai_client=async_openai_client)
            assistant = await mino.run_assistant()
            run_instance = await async_openai_client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant.id,
            )
        else:
            run_instance = await async_openai_client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
            )

        message_references = {}  # type: Dict[str, cl.Message]
        step_references = {}  # type: Dict[str, cl.Step]
        tool_outputs = []

        # Use context manager for safer execution
        async with managed_run_execution(thread_id, run_instance.id):
            # Periodically check for updates with a timeout for each operation
            while True:
                run_instance = await asyncio.wait_for(
                    async_openai_client.beta.threads.runs.retrieve(
                        thread_id=thread_id, run_id=run_instance.id
                    ),
                    timeout=30.0
                )

                # Fetch the run steps with timeout
                run_steps = await asyncio.wait_for(
                    async_openai_client.beta.threads.runs.steps.list(
                        thread_id=thread_id, run_id=run_instance.id, order="asc"
                    ),
                    timeout=30.0
                )

                for step in run_steps.data:
                    # Fetch step details with timeout
                    run_step = await asyncio.wait_for(
                        async_openai_client.beta.threads.runs.steps.retrieve(
                            thread_id=thread_id, run_id=run_instance.id, step_id=step.id
                        ),
                        timeout=30.0
                    )
                    step_details = run_step.step_details

                    # Process message creation
                    if step_details.type == "message_creation":
                        thread_message = await asyncio.wait_for(
                            async_openai_client.beta.threads.messages.retrieve(
                                message_id=step_details.message_creation.message_id,
                                thread_id=thread_id,
                            ),
                            timeout=30.0
                        )
                        await process_thread_message(message_references, thread_message, async_openai_client)

                    # Process tool calls
                    if step_details.type == "tool_calls":
                        for tool_call in step_details.tool_calls:
                            if isinstance(tool_call, dict):
                                tool_call = DictToObject(tool_call)

                            if tool_call.type == "code_interpreter":
                                await process_tool_call(
                                    step_references=step_references,
                                    step=step,
                                    tool_call=tool_call,
                                    name=tool_call.type,
                                    input=tool_call.code_interpreter.input
                                    or "# Generating code",
                                    output=tool_call.code_interpreter.outputs,
                                    show_input="python",
                                )

                                tool_outputs.append(
                                    {
                                        "output": tool_call.code_interpreter.outputs or "",
                                        "tool_call_id": tool_call.id,
                                    }
                                )

                            elif tool_call.type == "retrieval":
                                await process_tool_call(
                                    step_references=step_references,
                                    step=step,
                                    tool_call=tool_call,
                                    name=tool_call.type,
                                    input="Retrieving information",
                                    output="Retrieved information",
                                )

                            elif tool_call.type == "function":
                                function_name = tool_call.function.name
                                function_args = json.loads(tool_call.function.arguments)

                                function_output = tool_map[function_name](
                                    **json.loads(tool_call.function.arguments)
                                )

                                await process_tool_call(
                                    step_references=step_references,
                                    step=step,
                                    tool_call=tool_call,
                                    name=function_name,
                                    input=function_args,
                                    output=function_output,
                                    show_input="json",
                                )

                                tool_outputs.append(
                                    {"output": function_output, "tool_call_id": tool_call.id}
                                )

                # Submit tool outputs if required
                if (
                    run_instance.status == "requires_action"
                    and run_instance.required_action.type == "submit_tool_outputs"
                ):
                    await asyncio.wait_for(
                        async_openai_client.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread_id,
                            run_id=run_instance.id,
                            tool_outputs=tool_outputs,
                        ),
                        timeout=30.0
                    )

                # Wait between polling to reduce API load
                await asyncio.sleep(2)

                if run_instance.status in ["cancelled", "failed", "completed", "expired"]:
                    logger.info(f"Run {run_instance.id} finished with status: {run_instance.status}")
                    break

    except asyncio.TimeoutError:
        logger.error("Timeout occurred during run execution")
        await cl.Message(content="The operation timed out. Please try again with a simpler query.").send()
    except Exception as e:
        logger.error(f"Error in run: {e}")
        await handle_exception(e)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Handle incoming user messages and route them appropriately.

    Args:
        message: The user message object
    """
    try:
        if is_in_assistant_profile():
            thread = cl.user_session.get("thread")  # type: Thread
            files_ids = await process_files(message.elements, async_openai_client)
            await run(thread_id=thread.id, human_query=message.content, file_ids=files_ids)
        else:
            # Get message history
            messages = cl.user_session.get("message_history") or []
            
            # Check if current model is a reasoning model that benefits from <think> tags
            current_model = get_setting(conf.SETTINGS_CHAT_MODEL)
            is_reasoning = conf.is_reasoning_model(current_model)
            
            # If this is a reasoning model and <think> is not already in content, add it
            if is_reasoning and "<think>" not in message.content:
                # Clone the original message content
                original_content = message.content
                # Modify the message content to include <think> tag
                message.content = f"<think>{original_content}"
                logger.info(f"Automatically added <think> tag for reasoning model: {current_model}")

            if message.elements and len(message.elements) > 0:
                await handle_files_attachment(message, messages, async_openai_client)
            else:
                # Check for <think> tag directly in user request
                if "<think>" in message.content.lower():
                    logger.info("Processing message with <think> tag using thinking conversation handler")
                    await handle_thinking_conversation(message, messages, route_layer)
                else:
                    await handle_conversation(message, messages, route_layer)
    except asyncio.CancelledError:
        logger.warning("Message processing was cancelled")
        await cl.Message(content="The operation was cancelled. Please try again.").send()
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await handle_exception(e)


async def handle_thinking_conversation(
    message: cl.Message, messages: List[Dict[str, str]], route_layer: Any
) -> None:
    """
    Handles conversations with visible thinking process.
    Shows the AI's reasoning before presenting the final answer.

    This implementation exactly follows the pattern provided in the reference code.
    """
    # Get model and settings
    model = get_setting(conf.SETTINGS_CHAT_MODEL)
    temperature = get_setting(conf.SETTINGS_TEMPERATURE) or 0.7
    top_p = get_setting(conf.SETTINGS_TOP_P) or 0.9

    # Remove the <think> tag from the query
    query = message.content.replace("<think>", "").strip()
    update_message_history_from_user(query)

    # Track start time for the thinking duration
    start = time.time()

    try:
        # Create the stream
        stream = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. First use <think> tag to show your reasoning process, then close it with </think> and provide your final answer."},
                *messages,
                {"role": "user", "content": query}
            ],
            temperature=float(temperature),
            top_p=float(top_p),
            stream=True,
        )

        thinking = False

        # Streaming the thinking
        async with cl.Step(name="Thinking") as thinking_step:
            final_answer = cl.Message(content="", author=model)

            async for chunk in stream:
                delta = chunk.choices[0].delta

                # Only check content if it exists
                if delta.content:
                    # Check for exact <think> tag
                    if delta.content == "<think>":
                        thinking = True
                        continue

                    # Check for exact </think> tag
                    if delta.content == "</think>":
                        thinking = False
                        thought_for = round(time.time() - start)
                        thinking_step.name = f"Thought for {thought_for}s"
                        await thinking_step.update()
                        continue

                    # Route content based on thinking flag
                    if thinking:
                        await thinking_step.stream_token(delta.content)
                    else:
                        await final_answer.stream_token(delta.content)

        # Send final answer after completing the thinking step
        if final_answer.content:
            # Update message history
            update_message_history_from_assistant(final_answer.content)

            # Add TTS action if enabled
            enable_tts_response = get_setting(conf.SETTINGS_ENABLE_TTS_RESPONSE)
            if enable_tts_response:
                final_answer.actions = [
                    cl.Action(
                        icon="speech",
                        name="speak_chat_response_action",
                        payload={"value": final_answer.content},
                        tooltip="Speak response",
                        label="Speak response"
                    )
                ]

            await final_answer.send()
        else:
            # If no final answer was provided, create a fallback message
            await cl.Message(content="I've thought about this but don't have a specific answer to provide.", author=model).send()

    except asyncio.TimeoutError:
        logger.error(f"Timeout while processing chat completion with model {model}")
        await cl.Message(content="The operation timed out. Please try again with a shorter query.").send()
    except asyncio.CancelledError:
        logger.warning("Chat completion was cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in handle_thinking_conversation: {e}")
        await handle_exception(e)


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

        # Update all other settings
        setting_keys = [
            conf.SETTINGS_CHAT_MODEL,
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

        logger.info("Settings updated successfully")
    except Exception as e:
        logger.error(f"Error updating settings: {e}")


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
        logger.error(f"Error handling TTS response: {e}")
        await cl.Message(content="Failed to generate speech. Please try again.").send()