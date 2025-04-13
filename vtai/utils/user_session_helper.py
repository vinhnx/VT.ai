"""
User session helper utilities for the VT application.

Manages user session data and settings access.
"""

from typing import Any

import chainlit as cl

# Update imports to use vtai namespace
from vtai.utils.chat_profile import AppChatProfileType


def update_message_history_from_user(context: str) -> None:
    """
    Update message history with user content

    Args:
        context: Message content to add
    """
    update_message_history(context=context, role="user")


def update_message_history_from_assistant(context: str) -> None:
    """
    Update message history with assistant content

    Args:
        context: Message content to add
    """
    update_message_history(context=context, role="assistant")


def update_message_history(context: str, role: str) -> None:
    """
    Update message history with specified content and role

    Args:
        context: Message content to add
        role: Role of the message sender ('user' or 'assistant')
    """
    if not context or not role:
        return

    messages = cl.user_session.get("message_history") or []
    messages.append({"role": role, "content": context})
    cl.user_session.set("message_history", messages)


def get_user_session_id() -> str:
    """
    Get the current user session ID

    Returns:
        The user session ID string or empty string if not available
    """
    return cl.user_session.get("id") or ""


def get_setting(key: str) -> Any:
    """
    Retrieves a specific setting value from the user session

    Args:
        key: The settings key to retrieve

    Returns:
        The setting value or None if not found
    """
    settings = cl.user_session.get("chat_settings")
    if settings is None:
        return None

    return settings.get(key)


def is_in_assistant_profile() -> bool:
    """
    Determines if the current session is in Assistant profile mode

    Returns:
        True if in Assistant profile, False otherwise
    """
    chat_profile = cl.user_session.get("chat_profile")
    return chat_profile == AppChatProfileType.ASSISTANT.value
