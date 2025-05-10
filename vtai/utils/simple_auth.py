"""
Simple password-based authentication for VT.ai using Chainlit's built-in auth system.

This provides a basic authentication mechanism without requiring external services.
"""

from typing import Any, Dict, Optional

import chainlit as cl

from vtai.utils.config import logger


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """
    Basic authentication callback for Chainlit's password auth system.

    In a production environment, this would validate against a secure database
    with properly hashed passwords. For this prototype, we're using hardcoded credentials.

    Args:
        username: The username provided by the user
        password: The password provided by the user

    Returns:
        A cl.User object if authentication is successful, None otherwise
    """
    # Log authentication attempt (without the password)
    logger.info(f"Authentication attempt for user: {username}")

    # TODO: Replace with actual database check and proper password hashing
    # For prototype/demo purposes only - this should never be used in production
    if (username, password) == ("admin", "admin"):
        logger.info(f"User {username} successfully authenticated")
        return cl.User(
            identifier=username,
            metadata={
                "role": "admin",
                "provider": "credentials",
                # You can add additional user metadata here as needed
            },
        )
    # You can add additional user accounts here for testing
    elif (username, password) == ("user", "password"):
        logger.info(f"User {username} successfully authenticated")
        return cl.User(
            identifier=username,
            metadata={
                "role": "user",
                "provider": "credentials",
            },
        )
    else:
        logger.warning(f"Failed authentication attempt for user: {username}")
        return None


# Optional: Add helper functions for other auth-related tasks
def get_current_user_role() -> str:
    """
    Get the role of the currently authenticated user.

    Returns:
        The role of the current user, or 'guest' if not authenticated
    """
    user = cl.user_session.get("user")
    if user:
        return user.metadata.get("role", "user")
    return "guest"
