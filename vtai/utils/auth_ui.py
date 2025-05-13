"""
UI components for authentication status display.

This module provides UI elements to show the current authentication status and relevant user information.
"""

import asyncio
import traceback
from typing import Any, Dict, List, Optional

import chainlit as cl
from supabase import Client

from vtai.utils.config import logger


async def get_auth_actions() -> List[cl.Action]:
    """Returns a list of actions for login, sign-up, and OAuth authentication."""
    actions = [
        cl.Action(
            name="show_login_form",
            label="🔐 Login",
            description="Login with your email and password",
            payload={},  # Empty payload as it's required but not needed for this action
        ),
        cl.Action(
            name="show_signup_form",
            label="✍️ Sign Up",
            description="Create a new account",
            payload={},  # Empty payload as it's required but not needed for this action
        ),
    ]

    # Add OAuth buttons - these will automatically use the OAuth callback
    # No need to add handlers for these buttons as Chainlit will handle the OAuth flow
    actions.append(
        cl.Action(
            name="oauth_login_google",
            label="🔑 Login with Google",
            description="Login using your Google account",
            payload={"provider": "google"},
            url="/auth/oauth/google",
        )
    )

    return actions


async def handle_auth_error(
    error_message: str, exception: Optional[Exception] = None
) -> None:
    """Display an error message to the user and log the details.

    Args:
        error_message: The user-friendly error message to display
        exception: Optional exception object for detailed logging
    """
    if exception:
        logger.error(
            "Authentication error: %s - %s: %s",
            error_message,
            type(exception).__name__,
            str(exception),
        )
        logger.debug("Authentication error traceback: %s", traceback.format_exc())
    else:
        logger.error("Authentication error: %s", error_message)

    # Display error to the user
    await cl.Message(
        content=f"❌ **Authentication Error**\n\n{error_message}",
        author="System",
    ).send()

    # Add retry button
    auth_actions = await get_auth_actions()
    await cl.Message(
        content="Please try again or use a different authentication method:",
        author="Actions",
        actions=auth_actions,
    ).send()


async def display_auth_status(
    is_authenticated: bool, user_data: Optional[Any] = None
) -> None:
    """
    Displays the current authentication status in the sidebar.

    Args:
        is_authenticated: Whether the user is authenticated.
        user_data: The user data, either from Supabase (dict) or password auth (cl.User object).
    """
    if is_authenticated and user_data:
        # Handle different user object structures
        if isinstance(user_data, dict):
            # Supabase user data
            user_name = user_data.get("user_metadata", {}).get(
                "full_name", user_data.get("email", "User")
            )
            user_email = user_data.get("email", "No email available")
            user_id = user_data.get("id", "Unknown ID")

            # Determine auth provider based on identities if available
            identities = user_data.get("identities", [])
            if identities and len(identities) > 0:
                auth_provider = identities[0].get("provider", "Supabase")
                # Format provider name for display (capitalize)
                auth_provider = auth_provider.capitalize()
            else:
                auth_provider = "Supabase"
        else:
            # Chainlit User object (password auth or OAuth)
            user_name = user_data.identifier
            user_email = (
                user_data.metadata.get("email", "Local user")
                if hasattr(user_data, "metadata")
                else "Local user"
            )
            user_id = user_data.identifier

            if hasattr(user_data, "metadata") and user_data.metadata:
                # Check if this was an OAuth login
                if user_data.metadata.get("oauth", False):
                    provider = user_data.metadata.get("provider", "Unknown")
                    auth_provider = f"{provider.capitalize()} (OAuth)"
                else:
                    auth_provider = user_data.metadata.get("provider", "credentials")
                    # Format provider name for display
                    if auth_provider == "credentials":
                        auth_provider = "Password"
            else:
                auth_provider = "Unknown"
