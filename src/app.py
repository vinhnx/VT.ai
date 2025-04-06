"""
VT.ai - Main application entry point.

A multimodal AI chat application with dynamic conversation routing.
"""

import json
from typing import Any, Dict, List, Optional

import chainlit as cl
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
)
from utils.error_handlers import handle_exception
from utils.file_handlers import process_files
from utils.llm_profile_builder import build_llm_profile
from utils.media_processors import handle_tts_response
from utils.settings_builder import build_settings
from utils.user_session_helper import is_in_assistant_profile

# Initialize the application
route_layer, assistant_id = initialize_app()

# Initialize OpenAI client with better error handling
try:
    openai_client = OpenAI(max_retries=2)
    async_openai_client = AsyncOpenAI(max_retries=2)
    logger.info("Successfully initialized OpenAI clients")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI clients: {e}")
    raise

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
        thread = await async_openai_client.beta.threads.create()
        cl.user_session.set("thread", thread)


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
    init_message = await async_openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=human_query,
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
    
    # Periodically check for updates
    while True:
        run_instance = await async_openai_client.beta.threads.runs.retrieve(
            thread_id=thread_id, run_id=run_instance.id
        )

        # Fetch the run steps
        run_steps = await async_openai_client.beta.threads.runs.steps.list(
            thread_id=thread_id, run_id=run_instance.id, order="asc"
        )

        for step in run_steps.data:
            # Fetch step details
            run_step = await async_openai_client.beta.threads.runs.steps.retrieve(
                thread_id=thread_id, run_id=run_instance.id, step_id=step.id
            )
            step_details = run_step.step_details
            
            # Process message creation
            if step_details.type == "message_creation":
                thread_message = (
                    await async_openai_client.beta.threads.messages.retrieve(
                        message_id=step_details.message_creation.message_id,
                        thread_id=thread_id,
                    )
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
                await async_openai_client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run_instance.id,
                    tool_outputs=tool_outputs,
                )

        await cl.sleep(2)  # Refresh every 2 seconds
        if run_instance.status in ["cancelled", "failed", "completed", "expired"]:
            break


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

            if message.elements and len(message.elements) > 0:
                await handle_files_attachment(message, messages, async_openai_client)
            else:
                await handle_conversation(message, messages, route_layer)
    except Exception as e:
        await handle_exception(e)


@cl.on_settings_update
async def update_settings(settings: Dict[str, Any]) -> None:
    """
    Update user settings based on preferences.
    
    Args:
        settings: Dictionary of user settings
    """
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


@cl.action_callback("speak_chat_response_action")
async def on_speak_chat_response(action: cl.Action) -> None:
    """
    Handle TTS action triggered by the user.
    
    Args:
        action: The action object containing payload
    """
    await action.remove()
    value = action.payload.get("value") or ""
    await handle_tts_response(value, openai_client)