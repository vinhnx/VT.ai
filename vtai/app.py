"""
VT - Main application entry point.

A multimodal AI chat application with dynamic conversation routing.
"""

import os

import dotenv


def ensure_env_loaded():
    """Ensure .env is loaded for both main and Chainlit subprocesses."""
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_ANON_KEY"):
        dotenv.load_dotenv()


ensure_env_loaded()

import argparse
import asyncio
import atexit
import json
import shutil
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import chainlit as cl
from fastapi import HTTPException, Request, status
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from supabase import Client as SupabaseClient
from supabase import create_client

# Import only essential modules for startup
from vtai.utils import constants as const
from vtai.utils import llm_providers_config as conf
from vtai.utils.config import cleanup, initialize_app, load_model_prices, logger
from vtai.utils.error_handlers import handle_exception
from vtai.utils.settings_builder import build_settings
from vtai.utils.user_session_helper import get_setting

# Register cleanup function to ensure resources are properly released
atexit.register(cleanup)

# Initialize Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    logger.warning(
        "SUPABASE_URL and SUPABASE_ANON_KEY environment variables are not set. Authentication will not work."
    )
    supabase_client: Optional[SupabaseClient] = None
else:
    supabase_client: Optional[SupabaseClient] = create_client(
        SUPABASE_URL, SUPABASE_ANON_KEY
    )

# Flag to track if we're in fast mode
fast_mode = os.environ.get("VT_FAST_START") == "1"
if fast_mode:
    logger.info("Starting in fast mode - optimizing startup time")
    start_time = time.time()

# Initialize the application with improved client configuration
route_layer, _, openai_client, async_openai_client = initialize_app()


# Middleware for authentication
async def auth_middleware(request: Request, call_next):
    """
    FastAPI middleware to check for Supabase JWT and redirect if not authenticated.
    """
    logger.info(
        f"[AUTH_MIDDLEWARE] Path: {request.url.path} | supabase_client: {'set' if supabase_client else 'None'}"
    )

    # Define public paths where authentication is not required
    public_paths = [
        "/healthz",  # For main_app's health check
        "/public/",  # For main_app's own static files (e.g., from vtai/public directory)
        "/static-files/",  # Chainlit's primary static assets (JS, CSS)
        "/assets/",  # Potentially other assets served by Chainlit
        "/favicon.ico",  # Favicon for the Chainlit app
        "/ws",  # Chainlit's WebSocket connections (often /ws or /ws/socket.io/)
        # Add any other Chainlit-specific public paths if identified, e.g., for public file access
    ]

    # Allow /ws/socket.io/ for websockets explicitly
    if request.url.path.startswith("/ws/socket.io/"):
        logger.info(f"Path '{request.url.path}' is a WebSocket connection. Skipping auth for this specific path.")
        response = await call_next(request)
        return response

    matched_public_path = None
    for p_path in public_paths:
        if request.url.path.startswith(p_path):
            matched_public_path = p_path
            break

    if matched_public_path:
        logger.info(
            f"Path '{request.url.path}' is public because it starts with '{matched_public_path}'. Skipping auth."
        )
        response = await call_next(request)
        return response
    else:
        logger.info(
            f"Path '{request.url.path}' is NOT public. Proceeding with auth. (Checked against: {public_paths})"
        )

    if not supabase_client:
        logger.error("Supabase client not available. Authentication cannot proceed.")
        if "websocket" in request.scope.get("type", ""):
            pass
        return RedirectResponse(
            url=f"http://localhost:3000/auth/login?error=supabase_unavailable",
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        )

    token = request.cookies.get("supabase-auth-token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split("Bearer ")[1]

    user = None
    if token:
        try:
            user_response = supabase_client.auth.get_user(token)
            user = user_response.user
            if user:
                logger.info(
                    f"Authenticated user: {user.email if hasattr(user, 'email') else user}"
                )
        except Exception as e:
            logger.warning(f"Token validation failed: {e}")
            user = None
    if not user:
        login_url = "http://localhost:3000/auth/login"
        redirect_uri = quote(str(request.url), safe="")
        full_login_url = f"{login_url}?redirect_uri={redirect_uri}"
        logger.info(f"User not authenticated. Redirecting to {full_login_url}")
        return RedirectResponse(
            url=full_login_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT
        )

    response = await call_next(request)
    return response


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
    global config_chat_session

    # Import modules that are not needed during initial startup
    import audioop
    import subprocess

    import numpy as np

    from vtai.utils.conversation_handlers import (
        config_chat_session,
        handle_conversation,
        handle_files_attachment,
        handle_reasoning_conversation,
        handle_thinking_conversation,
    )
    from vtai.utils.dict_to_object import DictToObject
    from vtai.utils.file_handlers import process_files
    from vtai.utils.llm_profile_builder import build_llm_profile
    from vtai.utils.media_processors import handle_tts_response
    from vtai.utils.safe_execution import safe_execution
    from vtai.utils.starter_prompts import (
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

# Deprecated: Chainlit custom authentication. No longer used.


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


@cl.on_chat_start
async def start_chat():
    """
    Initialize the chat session with settings and system message.
    """
    start_time = time.time()

    # Load deferred imports now that the user has started a chat
    load_deferred_imports()

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


@cl.on_chat_start
def set_chainlit_user_on_auth():
    """Set user info in Chainlit session after successful authentication."""
    user = cl.user_session.get("user")
    if user:
        # Already set
        return
    # You should extract user info from your Supabase JWT/session here
    # For example, you might have user info in a cookie or session
    jwt_user = None
    try:
        jwt_user = cl.user_session.get("supabase_user")
    except Exception:
        pass
    if jwt_user:
        # Store user info in session for downstream use (no User class needed)
        cl.user_session.set("user", jwt_user)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Handle incoming user messages and route them appropriately.

    Args:
        message: The user message object
    """
    # Make sure all imports are loaded before processing messages
    load_deferred_imports()

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
            # Check if selected model supports LiteLLM's reasoning capabilities
            elif conf.supports_reasoning(current_model):
                logger.info(
                    f"Using enhanced reasoning with LiteLLM for model: {current_model}"
                )
                await handle_reasoning_conversation(message, messages, route_layer)
            else:
                await handle_conversation(message, messages, route_layer)
