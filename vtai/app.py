"""
VT - Main application entry point.

A multimodal AI chat application with dynamic conversation routing.
"""

import argparse
import asyncio
import atexit
import json
import os
import shutil
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import chainlit as cl
import dotenv
from supabase import Client, create_client  # Added Supabase

# Import only essential modules for startup
from utils import constants as const
from utils import llm_providers_config as conf
from utils.auth_handler import (
    check_authentication,
    create_test_mode_message,
    handle_password_login,
    handle_password_signup,
    initialize_auth,
    logout,
)
from utils.auth_ui import display_auth_status, get_auth_actions, show_subscription_info
from utils.config import cleanup, initialize_app, load_model_prices, logger
from utils.error_handlers import handle_exception
from utils.settings_builder import build_settings
from utils.user_session_helper import get_setting

# Register cleanup function to ensure resources are properly released
atexit.register(cleanup)

# Flag to track if we're in fast mode
fast_mode = os.environ.get("VT_FAST_START") == "1"
if fast_mode:
    logger.info("Starting in fast mode - optimizing startup time")
    start_time = time.time()

# Initialize the application with improved client configuration
route_layer, _, openai_client, async_openai_client = initialize_app()

# Explicitly load environment variables from .env
dotenv.load_dotenv(".env")

# Initialize Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning(
        "Supabase URL or Key not found in environment variables. "
        "Supabase integration will be disabled."
    )
    supabase_client: Optional[Client] = None
else:
    try:
        supabase_client: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        supabase_client = None

if fast_mode:
    logger.info(
        f"App initialization completed in {time.time() - start_time:.2f} seconds"
    )

# Support for deferred model prices loading
_model_prices_loaded = False
_imports_loaded = False


# Lazy import function to defer module importing
def load_deferred_imports():
    """Load modules only when needed to speed up initial startup"""
    global _imports_loaded

    if _imports_loaded:
        return

    import_start = time.time()

    # pylint: disable=global-statement
    global numpy, audioop, subprocess, build_llm_profile
    global process_files, handle_tts_response, safe_execution, get_command_route, get_command_template
    global set_commands, handle_conversation, handle_files_attachment, handle_thinking_conversation, handle_reasoning_conversation, DictToObject
    global config_chat_session, handle_deepseek_reasoner_with_logging

    # Import modules that are not needed during initial startup
    import audioop
    import subprocess

    import numpy as np
    from utils.conversation_handlers import (
        config_chat_session,
        handle_conversation,
        handle_files_attachment,
        handle_reasoning_conversation,
        handle_thinking_conversation,
    )
    from utils.deepseek_handler import handle_deepseek_reasoner_with_logging
    from utils.dict_to_object import DictToObject
    from utils.file_handlers import process_files
    from utils.llm_profile_builder import build_llm_profile
    from utils.media_processors import handle_tts_response
    from utils.safe_execution import safe_execution
    from utils.starter_prompts import (
        get_command_route,
        get_command_template,
        set_commands,
    )

    # Assign modules to global namespace
    numpy = np

    logger.debug(f"Deferred imports loaded in {time.time() - import_start:.2f} seconds")
    _imports_loaded = True


# Prepare model prices in background
def ensure_model_prices():
    """Ensure model prices are loaded when needed"""
    global _model_prices_loaded
    if not _model_prices_loaded:
        # Load model prices in the background when first needed
        load_model_prices()
        _model_prices_loaded = True


# App name constant
APP_NAME = const.APP_NAME


@cl.set_chat_profiles
async def build_chat_profile(_=None):
    """Define and set available chat profiles."""
    # Return profiles with the current chat profiles configuration
    return [
        cl.ChatProfile(
            name=profile.title,
            markdown_description=profile.description,
        )
        for profile in conf.APP_CHAT_PROFILES
    ]


@cl.action_callback("show_login_form")
async def on_show_login_form(action: cl.Action):
    await cl.Message(content="Preparing login form...").send()  # Feedback
    try:
        res = await cl.AskUserMessage(
            content="Please enter your email:", timeout=120, author="Login"
        ).send()
        if not res or not res.get("content"):
            await cl.Message(
                content="Login cancelled or timed out.", author="System"
            ).send()
            return
        email = res["content"].strip()

        res = await cl.AskUserMessage(
            content="Please enter your password:", timeout=120, author="Login"
        ).send()
        if not res or not res.get("content"):
            await cl.Message(
                content="Login cancelled or timed out.", author="System"
            ).send()
            return
        password = res["content"].strip()

        if supabase_client:
            user_data, error_message = await handle_password_login(
                supabase_client, email, password
            )
            if user_data:
                user_name = user_data.get("user_metadata", {}).get(
                    "full_name", user_data.get("email", "User")
                )
                await cl.Message(
                    content=f"✅ Welcome back, {user_name}! You are now logged in.",
                    author="System",
                ).send()
                # Refresh UI elements
                await cl.run_sync(cl.remove_actions_by_name("show_login_form"))
                await cl.run_sync(cl.remove_actions_by_name("show_signup_form"))
                await cl.Action(
                    name="logout_action",
                    label="Logout",
                    description="Sign out",
                    payload={},
                ).send(for_id="header")
                await display_auth_status(True, user_data)
                await show_subscription_info(supabase_client, user_data.get("id"))
            else:
                await cl.Message(
                    content=f"⚠️ Login Failed: {error_message}", author="System"
                ).send()
        else:
            await cl.Message(
                content="Auth service not available.", author="System"
            ).send()
    except Exception as e:
        logger.error(f"Error during login form: {e}")
        await cl.Message(
            content=f"An unexpected error occurred: {e}", author="System"
        ).send()


@cl.action_callback("show_signup_form")
async def on_show_signup_form(action: cl.Action):
    await cl.Message(content="Preparing sign-up form...").send()
    try:
        res = await cl.AskUserMessage(
            content="Please enter your email for sign-up:",
            timeout=120,
            author="Sign-up",
        ).send()
        if not res or not res.get("content"):
            await cl.Message(
                content="Sign-up cancelled or timed out.", author="System"
            ).send()
            return
        email = res["content"].strip()

        res = await cl.AskUserMessage(
            content="Please enter a password (min 6 characters):",
            timeout=120,
            author="Sign-up",
        ).send()
        if not res or not res.get("content"):
            await cl.Message(
                content="Sign-up cancelled or timed out.", author="System"
            ).send()
            return
        password = res["content"].strip()

        if supabase_client:
            user_data, message = await handle_password_signup(
                supabase_client, email, password
            )
            if user_data:  # Successfully signed up and logged in
                user_name = user_data.get("user_metadata", {}).get(
                    "full_name", user_data.get("email", "User")
                )
                await cl.Message(
                    content=f"✅ Welcome, {user_name}! You have successfully signed up and are logged in.",
                    author="System",
                ).send()
                # Refresh UI elements
                await cl.run_sync(cl.remove_actions_by_name("show_login_form"))
                await cl.run_sync(cl.remove_actions_by_name("show_signup_form"))
                await cl.Action(
                    name="logout_action",
                    label="Logout",
                    description="Sign out",
                    payload={},
                ).send(for_id="header")
                await display_auth_status(True, user_data)
                await show_subscription_info(supabase_client, user_data.get("id"))
            elif (
                message
            ):  # Signup successful, but action needed (e.g. email confirmation) or error
                await cl.Message(content=f"{message}", author="System").send()
            else:  # Should not happen if user_data is None and message is None, but as a fallback
                await cl.Message(
                    content="⚠️ Sign-up failed. Please try again.", author="System"
                ).send()
        else:
            await cl.Message(
                content="Auth service not available.", author="System"
            ).send()
    except Exception as e:
        logger.error(f"Error during signup form: {e}")
        await cl.Message(
            content=f"An unexpected error occurred: {e}", author="System"
        ).send()


@cl.action_callback("logout_action")
async def on_logout_action(_: cl.Action):
    """
    Callback for when the logout button is clicked.
    """
    # Check if user was authenticated via password auth
    user = cl.user_session.get("user")
    if (
        user
        and hasattr(user, "metadata")
        and user.metadata.get("provider") == "credentials"
    ):
        logger.info(f"Logging out password-authenticated user: {user.identifier}")
        # Reset session for password-authenticated users
        cl.user_session.set("user", None)
        cl.user_session.set("authenticated", False)
    else:
        # Handle Supabase logout
        await logout(supabase_client)

    # Reload the UI to update the session state
    # Remove logout button, add login/signup buttons
    await cl.run_sync(cl.remove_actions_by_name("logout_action"))
    auth_actions = await get_auth_actions()
    for auth_action in auth_actions:
        await auth_action.send(for_id="header")
    await display_auth_status(False)
    await create_test_mode_message()  # Show test mode message after logout


@cl.on_chat_start
async def start_chat():
    """
    Initialize the chat session with settings and system message.
    """
    start_time = time.time()

    # Load deferred imports now that the user has started a chat
    load_deferred_imports()

    # Check if user is authenticated via Chainlit's built-in password auth
    chainlit_user = cl.user_session.get("user")
    if (
        chainlit_user
        and hasattr(chainlit_user, "metadata")
        and chainlit_user.metadata.get("provider") == "credentials"
    ):
        logger.info(f"User authenticated via password auth: {chainlit_user.identifier}")
        cl.user_session.set("authenticated", True)

        # Display welcome message
        await cl.Message(
            content=f"Welcome, {chainlit_user.identifier}! You're logged in via password authentication.",
            author="System",
        ).send()

        # Display auth status and subscription info
        await display_auth_status(True, chainlit_user)
        await show_subscription_info(supabase_client, chainlit_user.identifier)

        # Add logout button to header
        await cl.Action(
            name="logout_action",
            label="Logout",
            description="Sign out of your account",
            payload={},
        ).send(for_id="header")
    # --- Initialize Supabase Auth ---
    elif supabase_client:
        logger.info("Supabase client is available, initializing authentication")
        await initialize_auth(supabase_client)
        is_authenticated, user_data = await check_authentication(supabase_client)

        if is_authenticated and user_data:
            # User is authenticated
            user_name = user_data.get("user_metadata", {}).get(
                "full_name", user_data.get("email", "User")
            )
            logger.info(f"User is authenticated via Supabase: {user_name}")
            await cl.Message(
                content=f"Welcome back, {user_name}! You're logged in.",
                author="System",
            ).send()

            # Display auth status in sidebar
            await display_auth_status(True, user_data)

            # Show subscription information
            await show_subscription_info(supabase_client, user_data.get("id"))

            # Add a logout button to the header
            await cl.Action(
                name="logout_action",
                label="Logout",
                description="Sign out of your account",
                payload={},
            ).send(for_id="header")
        else:
            # User is not authenticated - show login/signup buttons and test mode message
            logger.info("User is not authenticated, showing login/signup buttons")
            auth_actions = await get_auth_actions()
            for action in auth_actions:
                logger.info(f"Sending auth action: {action.name}")
                await action.send(for_id="header")
            await create_test_mode_message()

            # Display auth status for unauthenticated users
            await display_auth_status(False)
    else:
        logger.warning("Supabase client not initialized. Authentication is disabled.")
        cl.user_session.set("user", None)  # Default to no user
        cl.user_session.set("authenticated", False)

        # Even though Supabase client is not available, we still show login/signup buttons
        # for UI consistency and to prompt users to set up their credentials
        auth_actions = await get_auth_actions()
        for action in auth_actions:
            await action.send(for_id="header")

        await create_test_mode_message()  # Inform user about test mode if Supabase is down
        await display_auth_status(False)

    # Initialize default settings
    cl.user_session.set(conf.SETTINGS_CHAT_MODEL, conf.DEFAULT_MODEL)
    # Set default value for web search model
    cl.user_session.set(const.SETTINGS_WEB_SEARCH_MODEL, const.DEFAULT_WEB_SEARCH_MODEL)

    # Build LLM profile with direct icon path instead of using map
    build_llm_profile()

    # Settings configuration
    settings = await build_settings()

    # Configure chat session with selected model
    await config_chat_session(settings)

    # Initialize commands in the UI
    await set_commands(use_all=True)

    # Ensure model prices are loaded at this point
    ensure_model_prices()

    logger.debug(
        f"Chat session initialization completed in {time.time() - start_time:.2f} seconds"
    )


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Handle incoming user messages and route them appropriately.

    Args:
        message: The user message object
    """
    # Make sure all imports are loaded before processing messages
    load_deferred_imports()

    # --- Log User Info ---
    # First check if user is authenticated via password auth
    user = cl.user_session.get("user")
    if (
        user
        and hasattr(user, "metadata")
        and user.metadata.get("provider") == "credentials"
    ):
        # Password authenticated user
        user_id = user.identifier
        user_role = user.metadata.get("role", "user")
        logger.info(
            f"Processing message for password-authenticated user: {user_id} (role: {user_role})"
        )
    else:
        # Check Supabase authentication
        is_authenticated, user_info = await check_authentication(supabase_client)
        if is_authenticated and user_info:
            user_id = user_info.get("id", "Unknown")
            user_email = user_info.get("email", "No email")
            logger.info(
                f"Processing message for Supabase-authenticated user: {user_id} ({user_email})"
            )

            # In Phase 2, we would check subscription limits here
            # For now, every authenticated user has the same access
        else:
            logger.info("Processing message for anonymous user in test mode.")
    # --- End Log User Info ---

    async with safe_execution(
        operation_name="message processing",
        cancelled_message="The operation was cancelled. Please try again.",
    ):
        # Check if message has a command attached
        if message.command:
            logger.info(f"Processing message with command: {message.command}")
            # Get a template for the command if available
            template = get_command_template(message.command)

            # If this is a command without content, insert the template
            if not message.content.strip() and template:
                # Set the message content to the template
                message.content = template
                logger.info(f"Inserted template for command: {message.command}")

            # Get the route associated with this command
            route = get_command_route(message.command)
            if route:
                logger.info(f"Command {message.command} mapped to route: {route}")
                # Prepend route marker for better routing
                prefixed_content = f"[{route}] {message.content}"
                message.content = prefixed_content
            else:
                # Just prepend the command name if no specific route mapping is found
                prefixed_content = f"[{message.command}] {message.content}"
                message.content = prefixed_content

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
            # Check for DeepSeek Reasoner model
            elif current_model == "deepseek-reasoner":
                logger.info("Using DeepSeek Reasoner model with specialized handler")
                await handle_deepseek_reasoner_with_logging(
                    message, messages, route_layer
                )
            # Check if selected model supports LiteLLM's reasoning capabilities
            elif conf.supports_reasoning(current_model):
                logger.info(
                    f"Using enhanced reasoning with LiteLLM for model: {current_model}"
                )
                await handle_reasoning_conversation(message, messages, route_layer)
            else:
                await handle_conversation(message, messages, route_layer)
