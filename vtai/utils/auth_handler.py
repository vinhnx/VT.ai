"""
Supabase authentication handling for VT.ai.

Manages login, OAuth flows, and session management for user authentication.
"""

import asyncio
import os
from typing import Any, Dict, Optional, Tuple

import chainlit as cl
from supabase import Client

from vtai.utils.config import logger


async def initialize_auth(supabase_client: Optional[Client]) -> None:
    """
    Initializes authentication by checking for existing sessions and setting up necessary hooks.

    Args:
        supabase_client: The initialized Supabase client
    """
    if not supabase_client:
        logger.warning(
            "Supabase client not available. Auth functionality will be disabled."
        )
        cl.user_session.set("authenticated", False)  # Ensure defaults are set
        cl.user_session.set("user", None)
        return

    # Initialize user session defaults for this interaction
    cl.user_session.set("authenticated", False)
    cl.user_session.set("user", None)

    # Attempt to load and validate a stored session
    stored_session_data = cl.user_session.get("supabase_session_storage")

    if stored_session_data:
        try:
            logger.debug(
                f"Attempting to restore session for user if ID is present: {stored_session_data.get('user', {}).get('id')}"
            )
            # Restore session in Supabase client
            supabase_client.auth.set_session(
                access_token=stored_session_data["access_token"],
                refresh_token=stored_session_data["refresh_token"],
            )
            # Verify the session and get user details. This might also refresh the token.
            response = supabase_client.auth.get_user()

            if response and response.user:
                user_data = response.user.model_dump()
                # Get the latest session details (possibly refreshed)
                current_session_details = supabase_client.auth.get_session()
                if not current_session_details:
                    logger.warning(
                        "Failed to get current session details after get_user succeeded. Clearing stored session."
                    )
                    cl.user_session.set("supabase_session_storage", None)
                    # Keep authenticated as False, user as None (already set)
                    return

                cl.user_session.set("user", user_data)
                cl.user_session.set("authenticated", True)
                # Update the stored session with the latest (potentially refreshed) tokens
                cl.user_session.set(
                    "supabase_session_storage", current_session_details.model_dump()
                )

                logger.info(
                    f"User {user_data.get('id')} authenticated from stored session."
                )
                return  # Successfully authenticated
            else:
                logger.warning(
                    "Stored session found but get_user() failed. Clearing stored session."
                )
                cl.user_session.set("supabase_session_storage", None)

        except Exception as e:
            # This can happen if tokens are invalid, expired and unrefreshable, or other API errors.
            logger.warning(
                f"Failed to validate stored session: {e}. Clearing stored session."
            )
            cl.user_session.set("supabase_session_storage", None)
            # Ensure supabase client doesn't retain a bad session state from set_session attempt
            try:
                supabase_client.auth.sign_out()  # Clear any potentially lingering session in client
            except Exception:  # nosec
                pass  # Ignore errors during this cleanup sign_out

    # If no valid session was found or restored, authenticated remains False
    logger.debug("No valid stored session found or session restoration failed.")


async def handle_password_signup(
    supabase_client: Client, email: str, password: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Handles user sign-up with email and password.

    Args:
        supabase_client: The initialized Supabase client.
        email: User's email.
        password: User's password.

    Returns:
        A tuple (user_data, error_message). user_data is None if signup failed.
        error_message is None if signup was successful and user is logged in,
        or a message if email confirmation is needed.
    """
    try:
        response = supabase_client.auth.sign_up({"email": email, "password": password})
        if response.user and response.session:  # Successful signup, session created
            user_data = response.user.model_dump()
            session_data = response.session.model_dump()
            cl.user_session.set("user", user_data)
            cl.user_session.set("authenticated", True)
            cl.user_session.set(
                "supabase_session_storage", session_data
            )  # Persist this
            # Ensure client has session for immediate use
            supabase_client.auth.set_session(
                session_data["access_token"], session_data["refresh_token"]
            )
            logger.info(
                f"User {user_data.get('id')} signed up and logged in successfully."
            )
            return user_data, None
        elif (
            response.user and not response.session
        ):  # Signup successful, but email confirmation might be needed
            user_data = response.user.model_dump()
            logger.info(
                f"User {user_data.get('id')} signed up. Email confirmation may be required."
            )
            return (
                None,
                "Signup successful. Please check your email to confirm your account, then log in.",
            )
        else:
            logger.warning(
                f"Sign up attempt for {email} resulted in an unexpected response: {response}"
            )
            return None, "Sign up failed due to an unexpected issue. Please try again."

    except Exception as e:
        logger.error(f"Error during sign up for {email}: {e}")
        error_msg = str(e)
        if (
            "User already registered" in error_msg
            or "already exists" in error_msg.lower()
        ):  # More robust check
            return None, "This email is already registered. Please try logging in."
        if "Password should be at least 6 characters" in error_msg:
            return None, "Password should be at least 6 characters long."
        # Check for more specific Supabase/GoTrue error messages if available
        # For example, if e has a 'message' attribute from GoTrueApiError
        if hasattr(e, "message") and isinstance(e.message, str):
            if "rate limit exceeded" in e.message.lower():
                return None, "Sign up rate limit exceeded. Please try again later."
            return None, f"Sign up failed: {e.message}"
        return None, f"An error occurred during sign up. Please try again."


async def handle_password_login(
    supabase_client: Client, email: str, password: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Handles user login with email and password.

    Args:
        supabase_client: The initialized Supabase client.
        email: User's email.
        password: User's password.

    Returns:
        A tuple (user_data, error_message). user_data is None if login failed.
        error_message is None if login was successful.
    """
    try:
        response = supabase_client.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        if response.user and response.session:
            user_data = response.user.model_dump()
            session_data = response.session.model_dump()
            cl.user_session.set("user", user_data)
            cl.user_session.set("authenticated", True)
            cl.user_session.set(
                "supabase_session_storage", session_data
            )  # Persist this
            # Ensure client has session for immediate use
            supabase_client.auth.set_session(
                session_data["access_token"], session_data["refresh_token"]
            )
            logger.info(f"User {user_data.get('id')} logged in successfully.")
            return user_data, None
        else:
            # This case should ideally be caught by exceptions for invalid credentials.
            logger.warning(
                f"Login attempt for {email} failed with an unexpected response: {response}"
            )
            return (
                None,
                "Login failed. Please check your credentials or try again later.",
            )
    except Exception as e:
        logger.error(f"Error during login for {email}: {e}")
        error_msg = str(e)
        if "Invalid login credentials" in error_msg:
            return None, "Invalid email or password."
        if "Email not confirmed" in error_msg:
            return (
                None,
                "Email not confirmed. Please check your inbox for a confirmation link.",
            )
        # Check for more specific Supabase/GoTrue error messages
        if hasattr(e, "message") and isinstance(e.message, str):
            if "rate limit exceeded" in e.message.lower():
                return None, "Login rate limit exceeded. Please try again later."
            return None, f"Login failed: {e.message}"
        return None, f"An error occurred during login. Please try again."


async def create_test_mode_message() -> None:
    """
    Creates a message to inform the user they are in test mode.
    """
    await cl.Message(
        content="🧪 You're currently in **Test Mode**. Some features may be limited. "
        "Sign up or log in to access all features.",
        author="System",
    ).send()


async def get_user_subscription_tier(
    supabase_client: Client, user_id: str
) -> Optional[str]:
    """
    Retrieves the subscription tier for a user from Supabase.
    This is a placeholder for Phase 2 when subscription management is implemented.

    Args:
        supabase_client: The initialized Supabase client
        user_id: The user's ID

    Returns:
        The subscription tier name, or None if no active subscription exists
    """
    # This is a placeholder - in Phase 2, we'll implement actual subscription checks
    # For now, we'll just return a test tier
    return "basic_test"


async def check_authentication(
    supabase_client: Optional[
        Client
    ],  # Keep for API consistency, though direct use here is minimal
) -> Tuple[bool, Optional[Any]]:
    """
    Checks if the current user is authenticated based on Chainlit session state.
    Supports both Supabase authentication and password-based authentication.

    Args:
        supabase_client: The initialized Supabase client (currently unused here but kept for signature consistency).

    Returns:
        A tuple containing (is_authenticated, user_data)
    """
    # First check for password-based authentication
    user = cl.user_session.get("user")
    if (
        user
        and hasattr(user, "metadata")
        and user.metadata.get("provider") == "credentials"
    ):
        # User is authenticated via password auth
        return True, user

    # Then check for Supabase authentication
    is_authenticated = cl.user_session.get("authenticated", False)
    user_data = cl.user_session.get("user")

    if is_authenticated and user_data:
        # Assumes initialize_auth or login/signup handlers have correctly set up
        # the Supabase client's session state for the current user interaction.
        return True, user_data

    # If not authenticated or no user_data, ensure clean state
    if not is_authenticated:
        cl.user_session.set("user", None)  # Ensure user is None if not authenticated
        cl.user_session.set(
            "supabase_session_storage", None
        )  # Clear any stale session storage

    return False, None


async def logout(supabase_client: Optional[Client]) -> None:
    """
    Logs out the current user.
    Supports both Supabase and password-based authentication.

    Args:
        supabase_client: The initialized Supabase client
    """
    # First check for password-based authentication
    user = cl.user_session.get("user")
    if (
        user
        and hasattr(user, "metadata")
        and user.metadata.get("provider") == "credentials"
    ):
        # Handle password auth logout
        logger.info(f"Logging out password-authenticated user: {user.identifier}")
        cl.user_session.set("user", None)
        cl.user_session.set("authenticated", False)
        return

    # Handle Supabase authentication logout
    if supabase_client:
        try:
            current_session = supabase_client.auth.get_session()
            if (
                current_session
            ):  # Only attempt sign_out if there's a session in the client
                sign_out_response = supabase_client.auth.sign_out()
                # supabase-py v1's sign_out returns None on success, or an error object/raises error.
                if sign_out_response:
                    logger.warning(f"Supabase sign_out returned: {sign_out_response}")
            else:
                logger.debug("Logout called but no active session in Supabase client.")
        except Exception as e:
            logger.warning(f"Exception during Supabase sign out: {e}")

    # Clear local Chainlit session data
    cl.user_session.set("authenticated", False)
    cl.user_session.set("user", None)
    cl.user_session.set("supabase_session_storage", None)  # Clear stored session

    await cl.Message(
        content="👋 You have been successfully logged out.",
        author="System",
    ).send()
