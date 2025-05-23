"""
User session helper utilities for the VT application.

Manages user session data and settings access.
"""

from typing import Any, Dict, Optional

import chainlit as cl

# Note: Import of AppChatProfileType removed as assistant profile is no longer used


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


def get_user_profile() -> Dict[str, Any]:
    """
    Get the current user profile from the session.

    Returns:
        User profile dictionary or empty dict if not found
    """
    return cl.user_session.get("user_profile") or {}


def get_user_id() -> str:
    """
    Get the current user ID from the session.

    Returns:
        User ID string or empty string if not found
    """
    return cl.user_session.get("user_id") or ""


def get_user_email() -> str:
    """
    Get the current user email from the session.

    Returns:
        User email string or empty string if not found
    """
    return cl.user_session.get("user_email") or ""


def get_user_display_name() -> str:
    """
    Get the current user display name from the session.

    Returns:
        User display name string or empty string if not found
    """
    return cl.user_session.get("user_display_name") or ""


def get_chainlit_user() -> Optional[Any]:
    """
    Get the Chainlit User object from the session.

    Returns:
        Chainlit User object or None if not found
    """
    return cl.user_session.get("user")


# Note: is_in_assistant_profile() function has been removed as this feature is no longer supported
