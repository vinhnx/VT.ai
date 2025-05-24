import os
import time

import dotenv
import litellm


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
from vtai.utils.conversation_handlers import set_litellm_api_keys_from_settings
from vtai.utils.settings_builder import build_settings
from vtai.utils.supabase_logger import fetch_user_profile_from_supabase
from vtai.utils.user_session_helper import get_setting, get_user_profile

# Register cleanup function to ensure resources are properly released
atexit.register(cleanup)

# Optional JWT imports for enhanced security (not required for basic functionality)
try:
    from jose import jwt

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
subprocess = None
config_chat_session = None
handle_conversation = None
handle_files_attachment = None
DictToObject = None
process_files = None
build_llm_profile = None
handle_tts_response = None
safe_execution = None
get_command_route = None
get_command_template = None
set_commands = None


# Lazy import function to defer module importing
def load_deferred_imports() -> None:
    """Load modules only when needed to speed up initial startup."""
    # Globals required for lazy import pattern
    global _imports_loaded, numpy, subprocess
    global config_chat_session, handle_conversation, handle_files_attachment
    global DictToObject, process_files, build_llm_profile, handle_tts_response, safe_execution
    global get_command_route, get_command_template, set_commands
    if _imports_loaded:
        return
    import subprocess as sp

    import numpy as np

    from vtai.utils.conversation_handlers import (
        config_chat_session as _config_chat_session,
    )
    from vtai.utils.conversation_handlers import (
        handle_conversation as _handle_conversation,
    )
    from vtai.utils.conversation_handlers import (
        handle_files_attachment as _handle_files_attachment,
    )
    from vtai.utils.conversation_handlers import set_litellm_api_keys_from_settings
    from vtai.utils.conversation_handlers import (
        set_litellm_api_keys_from_settings as _set_litellm_api_keys_from_settings,
    )
    from vtai.utils.dict_to_object import DictToObject as _DictToObject
    from vtai.utils.file_handlers import process_files as _process_files
    from vtai.utils.llm_profile_builder import build_llm_profile as _build_llm_profile
    from vtai.utils.media_processors import handle_tts_response as _handle_tts_response
    from vtai.utils.safe_execution import safe_execution as _safe_execution
    from vtai.utils.starter_prompts import get_command_route as _get_command_route
    from vtai.utils.starter_prompts import get_command_template as _get_command_template
    from vtai.utils.starter_prompts import set_commands as _set_commands

    numpy = np
    subprocess = sp
    config_chat_session = _config_chat_session
    handle_conversation = _handle_conversation
    handle_files_attachment = _handle_files_attachment
    DictToObject = _DictToObject
    process_files = _process_files
    build_llm_profile = _build_llm_profile
    handle_tts_response = _handle_tts_response
    safe_execution = _safe_execution
    get_command_route = _get_command_route
    get_command_template = _get_command_template
    set_commands = _set_commands
    logger.debug("Deferred imports loaded.")
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

# Track emitted log events to prevent duplicates in this process
_emitted_log_events = set()


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

        log_key = f"user_session_{user_name}"
        if log_key not in _emitted_log_events:
            logger.info("User session initialized for %s via OAuth", user_name)
            _emitted_log_events.add(log_key)

        # Send welcome message with user information
        welcome_msg = f"""
👋 Hi **{user_name}**, welcome to VT.ai!

I'm here to help you brainstorm, answer questions, and get things done—just ask me anything.

✨ **Tip:** You can type `show profile` at any time to view your user profile and usage stats.

Need ideas? Try asking for a summary, code snippet, or creative suggestion!
"""
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

    if "chat_session_started" not in _emitted_log_events:
        logger.info("Chat session started with Chainlit OAuth authentication")
        _emitted_log_events.add("chat_session_started")
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

    content = message.content.strip()
    if content == "show profile":
        user_id = cl.user_session.get("user_id")
        profile = get_user_profile() or {}
        if not profile and user_id:
            profile = cl.run_sync(fetch_user_profile_from_supabase(user_id))
        if not profile:
            cl.run_sync(cl.Message(content="No user profile found.").send())
        else:
            cl.run_sync(
                cl.Message(
                    content="Your profile:",
                    elements=[cl.CustomElement(name="UserProfile", props=profile)],
                ).send()
            )
        return

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

        # Retrieve BYOK API keys from settings (plain text, no encryption)
        user_keys = {}
        for provider in [
            "openai",
            "anthropic",
            "gemini",
            "cohere",
            "mistral",
            "groq",
            "ollama",
            "deepseek",
            "openrouter",
            "lmstudio",
        ]:
            key_setting = f"byok_{provider}_api_key"
            api_key = get_setting(key_setting)
            if api_key:
                user_keys[provider] = api_key

        # --- Local model dynamic config ---
        current_model = get_setting(conf.SETTINGS_CHAT_MODEL)

        # Set litellm API keys from settings (plain text, no encryption)
        set_litellm_api_keys_from_settings(user_keys)

        if message.elements and len(message.elements) > 0:
            await handle_files_attachment(
                message, messages, async_openai_client, user_keys=user_keys
            )
        else:
            await handle_conversation(
                message, messages, route_layer, user_keys=user_keys
            )


def upsert_user_profile_from_oauth(
    provider_id: str, raw_user_data: Dict[str, str]
) -> Optional[Dict[str, str]]:
    """
    Upsert user profile in Supabase from OAuth data. Returns stored user row or None.
    """
    if not supabase_client:
        logger.error("Supabase client not available for OAuth callback")
        return None

    email = raw_user_data.get("email")
    name = raw_user_data.get("name") or raw_user_data.get("given_name", "")
    avatar_url = raw_user_data.get("picture")
    provider_user_id = raw_user_data.get("sub") or raw_user_data.get("id")

    if not email or not provider_user_id:
        logger.error("Missing required OAuth data: email or provider_user_id missing")
        return None

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
            logger.info("Found existing user profile for user_id: %s", user_id)
    except Exception as e:
        logger.error("Error checking existing user: %s: %s", type(e).__name__, str(e))

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
    if not existing_user:
        user_data.update(
            {"subscription_tier": "free", "tokens_used": 0, "created_at": "now()"}
        )
        logger.info("Creating new user profile for user_id: %s", user_id)

    try:
        upsert_response = (
            supabase_client.table("user_profiles")
            .upsert(user_data, on_conflict="user_id")
            .execute()
        )
        if upsert_response.data:
            logger.info("Successfully upserted user profile for user_id: %s", user_id)
            return upsert_response.data[0]
        else:
            logger.error("Failed to upsert user profile - no data returned")
            return None
    except Exception as e:
        logger.error("Error upserting user profile: %s: %s", type(e).__name__, str(e))
        return None


async def fetch_user_profile_from_supabase(user_id: str) -> dict:
    """Fetch user profile from Supabase database by user_id."""
    from vtai.utils.supabase_logger import supabase_client

    if not supabase_client:
        return {}
    try:
        response = (
            supabase_client.table("user_profiles")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )
        if response.data and len(response.data) > 0:
            return response.data[0]
        return {}
    except Exception as e:
        from vtai.utils.config import logger

        logger.error("Error fetching user profile: %s: %s", type(e).__name__, str(e))
        return {}


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    _token: str,  # noqa: ARG001 (unused but required by Chainlit)
    raw_user_data: Dict[str, str],
    _default_user: cl.User,  # noqa: ARG001 (unused but required by Chainlit)
) -> Optional[cl.User]:
    """
    Chainlit OAuth callback: upsert user profile and set name/avatar from OAuth.
    """
    logger.info("OAuth callback triggered for provider: %s", provider_id)
    logger.debug("Raw user data keys: %s", list(raw_user_data.keys()))
    # Do not log raw_user_data or any sensitive info
    stored_user = upsert_user_profile_from_oauth(provider_id, raw_user_data)
    if not stored_user:
        return None
    logger.info("OAuth avatar_url set for user_id: %s", stored_user.get("user_id"))
    return cl.User(
        identifier=stored_user["user_id"],
        display_name=stored_user.get("full_name") or stored_user["user_id"],
        avatar_url=stored_user.get("avatar_url"),  # Set avatar_url directly
        metadata={
            "email": stored_user.get("email"),
            "name": stored_user.get("full_name"),
            "avatar_url": stored_user.get("avatar_url"),
            "provider": stored_user.get("provider"),
            "provider_user_id": stored_user.get("provider_user_id"),
            "subscription_tier": stored_user.get("subscription_tier", "free"),
            "tokens_used": stored_user.get("tokens_used", 0),
            "created_at": stored_user.get("created_at"),
            "updated_at": stored_user.get("updated_at"),
            **raw_user_data,
        },
    )


@cl.action_callback("show_user_profile")
async def show_user_profile_action(action):
    """Show the current user's profile as a custom Chainlit element."""
    try:
        profile = get_user_profile()
        if not profile:
            await cl.Message(content="No user profile found.").send()
            return
        await cl.Message(
            content="Your profile:",
            elements=[cl.CustomElement(name="UserProfile", props=profile)],
        ).send()
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        await cl.Message(content="Failed to load user profile.").send()


@cl.action_callback("show_profile_action")
async def show_profile_action_handler(action):
    """Show the current user's profile as a custom Chainlit element (settings action)."""
    try:
        profile = get_user_profile()
        if not profile:
            await cl.Message(content="No user profile found.").send()
            return
        await cl.Message(
            content="Your profile:",
            elements=[cl.CustomElement(name="UserProfile", props=profile)],
        ).send()
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        await cl.Message(content="Failed to load user profile.").send()


@cl.action_callback("show_user_profile_select")
async def show_user_profile_select_action(action):
    """Show the current user's profile when the Select widget is set to 'Yes'."""
    try:
        if action.value != "Yes":
            return
        profile = get_user_profile()
        if not profile:
            await cl.Message(content="No user profile found.").send()
            return
        await cl.Message(
            content="Your profile:",
            elements=[cl.CustomElement(name="UserProfile", props=profile)],
        ).send()
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        await cl.Message(content="Failed to load user profile.").send()


@cl.on_settings_update
def on_settings_update(settings: dict) -> None:
    """Detect if the user set 'Show Profile' to 'Yes' in settings and show the profile."""
    if settings.get("show_profile_select") == "Yes":
        user_id = cl.user_session.get("user_id")
        profile = get_user_profile() or {}
        if not profile and user_id:
            profile = cl.run_sync(fetch_user_profile_from_supabase(user_id))
        if not profile:
            cl.run_sync(cl.Message(content="No user profile found.").send())
        else:
            cl.run_sync(
                cl.Message(
                    content="Your profile:",
                    elements=[cl.CustomElement(name="UserProfile", props=profile)],
                ).send()
            )
        # Reset the select to 'No' so user can trigger again
        settings["show_profile_select"] = "No"
