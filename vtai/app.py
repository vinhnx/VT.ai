"""
VT - Main application entry point.

A multimodal AI chat application with dynamic conversation routing.
"""

import argparse
import asyncio
import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import chainlit as cl
import dotenv
from chainlit.types import ChatProfile

# Import modules
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
from vtai.utils.settings_builder import build_settings
from vtai.utils.user_session_helper import get_setting, is_in_assistant_profile

# Initialize the application with improved client configuration
route_layer, assistant_id, openai_client, async_openai_client = initialize_app()

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

    # Since tools are temporarily removed, log and return placeholder
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
    if not assistant_id:
        # Log warning that we're using a placeholder as Mino is disabled
        logger.warning("No assistant ID provided and Mino assistant is disabled")
        raise ValueError("No assistant ID available. Please configure an assistant ID")
    else:
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
    try:
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
    except asyncio.CancelledError:
        logger.warning("Message processing was cancelled")
        await cl.Message(
            content="The operation was cancelled. Please try again."
        ).send()
    except (ValueError, KeyError, AttributeError) as e:
        logger.error("Error in message processing: %s", e)
        await handle_exception(e)
    except Exception as e:
        logger.error("Unexpected error processing message: %s", repr(e))
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


def setup_chainlit_config():
    """
    Sets up a centralized Chainlit configuration directory in ~/.config/vtai/.chainlit
    and creates symbolic links from the current directory to avoid file duplication.
    This process is fully automated and requires no user intervention.

    Returns:
        Path: Path to the centralized chainlit config directory
    """
    # Get the package installation directory
    pkg_dir = Path(__file__).parent.parent
    src_chainlit_dir = pkg_dir / ".chainlit"

    # Create centralized Chainlit config directory
    user_config_dir = Path(os.path.expanduser("~/.config/vtai"))
    chainlit_config_dir = user_config_dir / ".chainlit"

    # Create directory if it doesn't exist
    user_config_dir.mkdir(parents=True, exist_ok=True)
    chainlit_config_dir.mkdir(parents=True, exist_ok=True)

    # Check if we have a config.toml in the source package
    if src_chainlit_dir.exists() and src_chainlit_dir.is_dir():
        # Copy config.toml if it exists and target doesn't exist or is older
        src_config = src_chainlit_dir / "config.toml"
        dst_config = chainlit_config_dir / "config.toml"

        if src_config.exists() and (
            not dst_config.exists()
            or src_config.stat().st_mtime > dst_config.stat().st_mtime
        ):
            shutil.copy2(src_config, dst_config)
            logger.info(f"Copied default config.toml to {dst_config}")

        # Copy translations directory if it exists
        src_translations = src_chainlit_dir / "translations"
        dst_translations = chainlit_config_dir / "translations"

        if src_translations.exists() and src_translations.is_dir():
            # Create translations directory if it doesn't exist
            dst_translations.mkdir(exist_ok=True)

            # Copy translation files
            for trans_file in src_translations.glob("*.json"):
                dst_file = dst_translations / trans_file.name
                if (
                    not dst_file.exists()
                    or trans_file.stat().st_mtime > dst_file.stat().st_mtime
                ):
                    shutil.copy2(trans_file, dst_file)

            logger.info(f"Copied translations to {dst_translations}")

    # Handle the chainlit.md file - we'll create an empty one to prevent Chainlit from creating its own
    src_md = pkg_dir / "chainlit.md"
    central_md = user_config_dir / "chainlit.md"

    # If our source package has a custom chainlit.md, use that, otherwise create empty file
    if src_md.exists() and src_md.stat().st_size > 0:
        # Copy the non-empty chainlit.md file to the central location
        if (
            not central_md.exists()
            or src_md.stat().st_mtime > central_md.stat().st_mtime
        ):
            shutil.copy2(src_md, central_md)
            logger.info(f"Copied custom chainlit.md to {central_md}")
    else:
        # Create an empty chainlit.md file in the central location if it doesn't exist
        if not central_md.exists():
            central_md.touch()
            logger.info(f"Created empty chainlit.md at {central_md}")

    # Create symbolic links in current directory to point to the central config
    current_dir = Path.cwd()
    local_chainlit_dir = current_dir / ".chainlit"
    local_md = current_dir / "chainlit.md"

    # Don't create symlinks when we're already in the project directory
    if str(current_dir) != str(pkg_dir.parent):
        try:
            # Handle .chainlit directory
            if local_chainlit_dir.exists():
                if local_chainlit_dir.is_symlink():
                    # If it's already a symlink, no need to do anything
                    pass
                else:
                    # If it's a regular directory, rename it as backup
                    backup_dir = current_dir / ".chainlit.backup"
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)
                    local_chainlit_dir.rename(backup_dir)
                    os.symlink(str(chainlit_config_dir), str(local_chainlit_dir))
                    logger.info(
                        f"Created symlink from {local_chainlit_dir} to {chainlit_config_dir}"
                    )
            else:
                # Create a symlink to the centralized config
                os.symlink(str(chainlit_config_dir), str(local_chainlit_dir))
                logger.info(
                    f"Created symlink from {local_chainlit_dir} to {chainlit_config_dir}"
                )

            # Handle chainlit.md file - CRITICAL: Create this BEFORE Chainlit starts
            # Create an empty chainlit.md in the current directory to prevent Chainlit from creating one
            if not local_md.exists():
                # Just create an empty file (not a symlink) to prevent Chainlit's default content
                local_md.touch()
                logger.info(
                    f"Created empty chainlit.md at {local_md} to prevent default content"
                )
        except Exception as e:
            logger.warning(
                f"Could not create symlinks or empty files: {e}. Using local files instead."
            )
            # If symlink creation fails, ensure the local directory exists
            local_chainlit_dir.mkdir(exist_ok=True)

            # Create an empty chainlit.md file if it doesn't exist
            if not local_md.exists():
                try:
                    local_md.touch()
                    logger.info(f"Created empty chainlit.md at {local_md} as fallback")
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
