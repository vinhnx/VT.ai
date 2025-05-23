"""
VT - Main application entry p# Register cleanup function to ensure resources are properly released
atexit.register(cleanup)ltimodal AI chat application with dynamic conversation routing.
"""

import os
import time

import dotenv


def ensure_env_loaded():
    """Ensure .env is loaded for both main and Chainlit subprocesses."""
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_ANON_KEY"):
        dotenv.load_dotenv()


ensure_env_loaded()

import atexit
from typing import Dict, Optional

import chainlit as cl
from supabase import Client as SupabaseClient
from supabase import create_client

from vtai.utils import constants as const
from vtai.utils import llm_providers_config as conf
from vtai.utils.config import cleanup, initialize_app, load_model_prices, logger
from vtai.utils.settings_builder import build_settings
from vtai.utils.user_session_helper import get_setting

# Register cleanup function to ensure resources are properly released
atexit.register(cleanup)

# Optional JWT imports for enhanced security (not required for basic functionality)
try:
    from jose import JWTError, jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("python-jose not available - some JWT features may be limited")


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


if fast_mode:
    logger.info(
        "App initialization completed in %.2f seconds", time.time() - start_time
    )

# Support for deferred model prices loading
_model_prices_loaded = False
_imports_loaded = False

# Global variables for deferred imports
numpy = None
audioop = None
subprocess = None
build_llm_profile = None
process_files = None
handle_tts_response = None
safe_execution = None
get_command_route = None
get_command_template = None
set_commands = None
handle_conversation = None
handle_files_attachment = None
handle_thinking_conversation = None
handle_reasoning_conversation = None
DictToObject = None
config_chat_session = None


# Lazy import function to defer module importing
def load_deferred_imports():
    """Load modules only when needed to speed up initial startup"""
    global _imports_loaded
    global numpy, audioop, subprocess, build_llm_profile
    global process_files, handle_tts_response, safe_execution, get_command_route, get_command_template
    global set_commands, handle_conversation, handle_files_attachment, handle_thinking_conversation, handle_reasoning_conversation, DictToObject
    global config_chat_session

    if _imports_loaded:
        return

    import_start = time.time()

    # Import modules that are not needed during initial startup
    import audioop as _audioop
    import subprocess as _subprocess

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
    audioop = _audioop
    subprocess = _subprocess

    logger.debug("Deferred imports loaded in %.2f seconds", time.time() - import_start)
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

# Note: Removed @cl.header_auth_callback to prevent infinite redirect loop
# Authentication is handled by FastAPI middleware, so we trust that validation for Chainlit


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
async def chainlit_chat_start():
    """Initialize chat session settings and user profile management."""
    start_time = time.time()

    # Load deferred imports
    load_deferred_imports()

    # Get current authenticated user from Chainlit OAuth
    current_user = cl.user_session.get("user")
    if current_user:
        # Extract user information from Chainlit user metadata
        user_metadata = current_user.metadata or {}
        user_email = user_metadata.get("email", "Unknown")
        user_name = user_metadata.get("name", user_email.split("@")[0])

        # Store user information in Chainlit session
        cl.user_session.set("user_id", current_user.identifier)
        cl.user_session.set("user_email", user_email)
        cl.user_session.set("user_display_name", user_name)
        cl.user_session.set("user_metadata", user_metadata)

        logger.info(
            "User session initialized for %s (%s) via OAuth", user_name, user_email
        )

        # Send welcome message with user information
        welcome_msg = f"👋 Welcome, **{user_name}**! How can I help you today?"
        await cl.Message(content=welcome_msg).send()
    else:
        logger.warning(
            "No authenticated user found - OAuth authentication may have failed"
        )

    # Initialize default settings
    cl.user_session.set(conf.SETTINGS_CHAT_MODEL, conf.DEFAULT_MODEL)
    cl.user_session.set(const.SETTINGS_WEB_SEARCH_MODEL, const.DEFAULT_WEB_SEARCH_MODEL)

    # Build LLM profile
    build_llm_profile()

    # Settings configuration
    settings = await build_settings()
    await config_chat_session(settings)
    await set_commands(use_all=True)

    # Ensure model prices are loaded
    ensure_model_prices()

    logger.info("Chat session started with Chainlit OAuth authentication")
    logger.debug(
        "Chat session initialization completed in %.2f seconds",
        time.time() - start_time,
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

    async with safe_execution(
        operation_name="message processing",
        cancelled_message="The operation was cancelled. Please try again.",
    ):
        # Check if message has a command attached
        if message.command:
            logger.info("Processing message with command: %s", message.command)
            # Get a template for the command if available
            template = get_command_template(message.command)

            # If this is a command without content, insert the template
            if not message.content.strip() and template:
                # Set the message content to the template
                message.content = template
                logger.info("Inserted template for command: %s", message.command)

            # Get the route associated with this command
            route = get_command_route(message.command)
            if route:
                logger.info("Command %s mapped to route: %s", message.command, route)
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
                    "Processing message with <think> tag using thinking conversation handler"
                )
                await handle_thinking_conversation(message, messages, route_layer)
            # Check if selected model supports LiteLLM's reasoning capabilities
            elif conf.supports_reasoning(current_model):
                logger.info(
                    "Using enhanced reasoning with LiteLLM for model: %s",
                    current_model,
                )
                await handle_reasoning_conversation(message, messages, route_layer)
            else:
                await handle_conversation(message, messages, route_layer)


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,  # noqa: ARG001 (unused but required by Chainlit)
    raw_user_data: Dict[str, str],
    default_user: cl.User,  # noqa: ARG001 (unused but required by Chainlit)
) -> Optional[cl.User]:
    """
    Handle OAuth authentication callback and manage user data in Supabase.

    Args:
        provider_id: OAuth provider (e.g., 'google')
        token: OAuth access token (unused but required by Chainlit)
        raw_user_data: Raw user data from OAuth provider
        default_user: Default Chainlit user object (unused but required by Chainlit)

    Returns:
        cl.User object if authentication successful, None otherwise
    """
    try:
        logger.info("OAuth callback triggered for provider: %s", provider_id)
        logger.debug("Raw user data keys: %s", list(raw_user_data.keys()))

        if not supabase_client:
            logger.error("Supabase client not available for OAuth callback")
            return None

        # Extract user information from OAuth data
        email = raw_user_data.get("email")
        name = raw_user_data.get("name") or raw_user_data.get("given_name", "")
        avatar_url = raw_user_data.get("picture")
        provider_user_id = raw_user_data.get("sub") or raw_user_data.get("id")

        if not email or not provider_user_id:
            logger.error(
                "Missing required OAuth data: email=%s, provider_user_id=%s",
                email,
                provider_user_id,
            )
            return None

        logger.info("Processing OAuth login for %s via %s", email, provider_id)

        # Create unique user_id combining provider and provider_user_id
        user_id = f"{provider_id}_{provider_user_id}"

        # Try to get existing user from user_profiles table
        existing_user = None
        try:
            response = (
                supabase_client.table("user_profiles")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )
            if response.data:
                existing_user = response.data[0]
                logger.info("Found existing user: %s", email)
        except Exception as e:
            logger.warning("Error checking existing user: %s", e)

        # Prepare user data for upsert
        user_data = {
            "user_id": user_id,
            "email": email,
            "full_name": name,
            "avatar_url": avatar_url,
            "provider": provider_id,
            "provider_user_id": provider_user_id,
            "raw_oauth_details": raw_user_data,
            "updated_at": "now()",
        }

        # If new user, set creation timestamp and default values
        if not existing_user:
            user_data.update(
                {"subscription_tier": "free", "tokens_used": 0, "created_at": "now()"}
            )
            logger.info("Creating new user profile for %s", email)

        # Upsert user data to Supabase
        try:
            upsert_response = (
                supabase_client.table("user_profiles")
                .upsert(user_data, on_conflict="user_id")
                .execute()
            )

            if upsert_response.data:
                logger.info("Successfully upserted user profile for %s", email)
                stored_user = upsert_response.data[0]
            else:
                logger.error("Failed to upsert user profile - no data returned")
                return None

        except Exception as e:
            logger.error("Error upserting user profile: %s", e)
            return None

        # Create and return Chainlit User object
        chainlit_user = cl.User(
            identifier=user_id,
            metadata={
                "email": email,
                "name": name,
                "avatar_url": avatar_url,
                "provider": provider_id,
                "provider_user_id": provider_user_id,
                "subscription_tier": stored_user.get("subscription_tier", "free"),
                "tokens_used": stored_user.get("tokens_used", 0),
                "created_at": stored_user.get("created_at"),
                "updated_at": stored_user.get("updated_at"),
                **raw_user_data,  # Include all raw OAuth data
            },
        )

        logger.info("OAuth authentication successful for %s (%s)", email, user_id)
        return chainlit_user

    except Exception as e:
        logger.error("Error in OAuth callback: %s: %s", type(e).__name__, str(e))
        return None
