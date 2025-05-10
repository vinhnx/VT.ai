"""
Authentication handling for VT.ai.

Manages both Supabase and simple password-based authentication, including login,
OAuth flows, session management, password reset, and email verification.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import chainlit as cl
from gotrue.errors import AuthApiError, AuthUnknownError
from supabase import Client
from supabase.lib.client_options import ClientOptions

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
    logger.info("Authentication attempt for user: %s", username)

    # TODO: Replace with actual database check and proper password hashing
    # For prototype/demo purposes only - this should never be used in production
    if (username, password) == ("admin", "admin"):
        logger.info("User %s successfully authenticated", username)
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
        logger.info("User %s successfully authenticated", username)
        return cl.User(
            identifier=username,
            metadata={
                "role": "user",
                "provider": "credentials",
            },
        )
    else:
        logger.warning("Failed authentication attempt for user: %s", username)
        return None


@cl.oauth_callback
async def oauth_callback(
    provider_id: str, token: str, raw_user_data: Dict[str, Any], default_user: cl.User
) -> Optional[cl.User]:
    """
    OAuth callback for Chainlit's OAuth authentication system.

    This function is called when a user successfully authenticates via OAuth.
    It enhances the default user object with information from the provider.

    Args:
        provider_id: The OAuth provider ID (e.g., 'google', 'github', etc.)
        token: The OAuth access token (use with care, avoid excessive logging)
        raw_user_data: The raw user data from the OAuth provider
        default_user: The default Chainlit user object created by Chainlit

    Returns:
        A cl.User object if authentication is successful, None otherwise
    """
    logger.info("-----------------------------------------------------")
    logger.info("ENTERING OAUTH_CALLBACK (REFINED)")
    logger.info(f"Provider ID: {provider_id}")
    logger.info(
        f"Token: {token[:15]}... (truncated for security)"
    )  # Log only a small part
    logger.debug(f"Raw User Data: {raw_user_data}")  # Full data only in debug
    logger.info(
        f"Default User (incoming): Identifier: {default_user.identifier}, Metadata: {default_user.metadata}"
    )
    logger.info("-----------------------------------------------------")

    email: Optional[str] = None
    name: Optional[str] = None
    picture: Optional[str] = None
    user_provider_specific_id: Optional[str] = None

    if provider_id == "google":
        user_provider_specific_id = raw_user_data.get("sub")
        email = raw_user_data.get("email")
        name = raw_user_data.get("name")
        picture = raw_user_data.get("picture")
    elif provider_id == "github":
        gh_id = raw_user_data.get("id")
        user_provider_specific_id = str(gh_id) if gh_id is not None else None
        email = raw_user_data.get("email")  # May be null
        name = raw_user_data.get("name") or raw_user_data.get("login")
        picture = raw_user_data.get("avatar_url")
    else:
        logger.warning(f"OAuth callback for unhandled provider: {provider_id}")
        # For unhandled providers, we'll rely on default_user and whatever Chainlit populates.

    # Use default_user as the base and enhance its metadata.
    # Chainlit typically sets default_user.identifier (e.g., "google:user_id_hash").
    # We will primarily enhance its metadata.
    enhanced_user = default_user

    if enhanced_user.metadata is None:  # Ensure metadata dict exists
        enhanced_user.metadata = {}

    # Update metadata with standardized and provider-specific info
    enhanced_user.metadata["provider"] = provider_id
    enhanced_user.metadata["oauth"] = True  # Mark as OAuth user
    if email:
        enhanced_user.metadata["email"] = email
    if name:
        # Use provider name, potentially overwriting if default_user had one from a generic source
        enhanced_user.metadata["name"] = name
    if picture:
        enhanced_user.metadata["picture"] = picture
    if user_provider_specific_id:
        enhanced_user.metadata["user_provider_id"] = user_provider_specific_id
    # Store a subset or all of raw_user_data if needed for debugging or other features,
    # but be mindful of size and sensitivity.
    # enhanced_user.metadata["raw_oauth_data"] = raw_user_data # Example

    logger.info(
        f"Enhanced User prepared: Identifier: {enhanced_user.identifier}, Metadata: {enhanced_user.metadata}"
    )

    # Attempt to set our custom session variables.
    # This is where the "Chainlit context not found" error might occur.
    # Even if this fails, returning enhanced_user allows Chainlit to proceed.
    try:
        logger.info("Attempting to set user session variables...")
        cl.user_session.set("user", enhanced_user)  # Store the cl.User object
        cl.user_session.set("authenticated", True)
        cl.user_session.set("auth_provider_type", "oauth")
        cl.user_session.set("oauth_provider_id", provider_id)  # Store specific provider
        logger.info("User session variables set successfully in oauth_callback.")
    except Exception as e:
        logger.error(
            f"Error setting user session variables in OAuth callback: {type(e).__name__}: {e}"
        )
        logger.exception("Full traceback for session setting error in oauth_callback:")
        # Continue without these custom session variables if an error occurs.
        # Chainlit's own session management based on the returned user should still function.

    logger.info("-----------------------------------------------------")
    logger.info("EXITING OAUTH_CALLBACK (REFINED) - RETURNING ENHANCED USER")
    logger.info("-----------------------------------------------------")
    return enhanced_user


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


async def initialize_auth(supabase_client: Optional[Client]) -> None:
    """
    Initializes authentication by checking for existing sessions and setting up necessary hooks.
    Supports both Supabase and simple password-based authentication.

    Args:
            supabase_client: The initialized Supabase client
    """
    if not supabase_client:
        logger.warning(
            "Supabase client not available. Only simple password auth will be available."
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
            user_id = stored_session_data.get("user", {}).get("id")
            logger.debug(
                "Attempting to restore session for user if ID is present: %s", user_id
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
                    "User %s authenticated from stored session.", user_data.get("id")
                )
                return  # Successfully authenticated
            else:
                logger.warning(
                    "Stored session found but get_user() failed. Clearing stored session."
                )
                cl.user_session.set("supabase_session_storage", None)

        except AuthApiError as e:
            # This can happen if tokens are invalid or expired
            logger.warning(
                "Failed to validate stored session due to auth API error: %s. Clearing stored session.",
                str(e),
            )
            cl.user_session.set("supabase_session_storage", None)
            # Ensure supabase client doesn't retain a bad session state from set_session attempt
            try:
                supabase_client.auth.sign_out()  # Clear any potentially lingering session in client
            except Exception:  # nosec
                pass  # Ignore errors during this cleanup sign_out

        except AuthUnknownError as e:
            # Handle authentication errors separately
            logger.warning(
                "Failed to validate stored session due to auth unknown error: %s. Clearing stored session.",
                str(e),
            )
            cl.user_session.set("supabase_session_storage", None)
            try:
                supabase_client.auth.sign_out()
            except Exception:  # nosec
                pass

        except Exception as e:
            # Handle other unexpected errors
            logger.warning(
                "Failed to validate stored session: %s. Clearing stored session.",
                str(e),
            )
            cl.user_session.set("supabase_session_storage", None)
            try:
                supabase_client.auth.sign_out()  # Clear any potentially lingering session in client
            except Exception:  # nosec
                pass  # Ignore errors during this cleanup sign_out

    # If no valid session was found or restored, authenticated remains False
    logger.debug("No valid stored session found or session restoration failed.")


async def handle_password_signup(
    supabase_client: Client,
    email: str,
    password: str,
    user_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Handles user sign-up with email and password.

    Args:
        supabase_client: The initialized Supabase client.
        email: User's email.
        password: User's password.
        user_metadata: Optional user metadata to store with the user.

    Returns:
        A tuple (user_data, error_message). user_data is None if signup failed.
        error_message is None if signup was successful and user is logged in,
        or a message if email confirmation is needed.
    """
    try:
        # Sign up user according to the auth-py format
        response = supabase_client.auth.sign_up(
            {
                "email": email,
                "password": password,
                "options": {
                    "data": user_metadata or {},
                },
            }
        )

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
                "User %s signed up and logged in successfully.", user_data.get("id")
            )
            return user_data, None
        elif (
            response.user and not response.session
        ):  # Signup successful, but email confirmation might be needed
            user_data = response.user.model_dump()
            logger.info(
                "User %s signed up. Email confirmation may be required.",
                user_data.get("id"),
            )
            return (
                None,
                "Signup successful. Please check your email to confirm your account, then log in.",
            )
        else:
            logger.warning(
                "Sign up attempt for %s resulted in an unexpected response: %s",
                email,
                response,
            )
            return None, "Sign up failed due to an unexpected issue. Please try again."

    except AuthApiError as e:
        logger.error("Auth API error during sign up for %s: %s", email, str(e))
        if "User already registered" in str(e) or "already exists" in str(e).lower():
            return None, "This email is already registered. Please try logging in."
        if "Password should be at least" in str(e):
            return None, f"Password error: {e.message}"
        if "rate limit" in str(e).lower():
            return None, "Sign up rate limit exceeded. Please try again later."
        return None, f"Sign up failed: {e.message}"

    except AuthUnknownError as e:
        logger.error("Auth unknown error during sign up for %s: %s", email, str(e))
        return None, f"Sign up failed due to an authentication error: {e.message}"

    except Exception as e:
        logger.error("Error during sign up for %s: %s", email, str(e))
        error_msg = str(e)
        if (
            "User already registered" in error_msg
            or "already exists" in error_msg.lower()
        ):
            return None, "This email is already registered. Please try logging in."
        if "Password should be at least" in error_msg:
            return None, "Password should be at least 6 characters long."
        return None, f"An error occurred during sign up: {error_msg}"


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
        # Login using the auth-py format
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
            logger.info("User %s logged in successfully.", user_data.get("id"))
            return user_data, None
        else:
            # This case should ideally be caught by exceptions for invalid credentials.
            logger.warning(
                "Login attempt for %s failed with an unexpected response: %s",
                email,
                response,
            )
            return (
                None,
                "Login failed. Please check your credentials or try again later.",
            )

    except AuthApiError as e:
        logger.error("Auth API error during login for %s: %s", email, str(e))
        if "Invalid login credentials" in str(e):
            return None, "Invalid email or password."
        if "Email not confirmed" in str(e):
            return (
                None,
                "Email not confirmed. Please check your inbox for a confirmation link.",
            )
        if "rate limit" in str(e).lower():
            return None, "Login rate limit exceeded. Please try again later."
        return None, f"Login failed: {e.message}"

    except AuthUnknownError as e:
        logger.error("Auth unknown error during login for %s: %s", email, str(e))
        return None, f"Login failed due to an authentication error: {e.message}"

    except Exception as e:
        logger.error("Error during login for %s: %s", email, str(e))
        error_msg = str(e)
        if "Invalid login credentials" in error_msg:
            return None, "Invalid email or password."
        if "Email not confirmed" in error_msg:
            return (
                None,
                "Email not confirmed. Please check your inbox for a confirmation link.",
            )
        return None, f"An error occurred during login: {error_msg}"


async def handle_oauth_signup(
    supabase_client: Client, provider: str, redirect_to: Optional[str] = None
) -> Dict[str, Any]:
    """Initiates the OAuth sign-in flow with a third-party provider.

    Args:
        supabase_client: The initialized Supabase client.
        provider: The OAuth provider to use (e.g., 'google', 'github', etc.).
        redirect_to: Optional URL to redirect to after successful sign-in.

    Returns:
        A dictionary containing the OAuth sign-in URL.
    """
    try:
        # Initiate OAuth flow according to auth-py format
        response = supabase_client.auth.sign_in_with_oauth(
            {
                "provider": provider,
                "options": {
                    "redirect_to": redirect_to,
                },
            }
        )

        logger.info("Initiated OAuth flow for provider: %s", provider)

        # Return the URL that the user should be redirected to
        return {"provider": response.provider, "url": response.url}

    except AuthApiError as e:
        logger.error("Auth API error during OAuth flow with %s: %s", provider, str(e))
        return {"error": f"Failed to initiate OAuth flow: {e.message}"}

    except AuthUnknownError as e:
        logger.error(
            "Auth unknown error during OAuth flow with %s: %s", provider, str(e)
        )
        return {
            "error": f"Failed to initiate OAuth flow due to an authentication error: {e.message}"
        }

    except Exception as e:
        logger.error("Error initiating OAuth flow with %s: %s", provider, str(e))
        if hasattr(e, "message") and isinstance(e.message, str):
            error_message = e.message
        else:
            error_message = str(e)

        return {"error": f"Failed to initiate OAuth flow: {error_message}"}


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
        and isinstance(user, cl.User)
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
        and isinstance(user, cl.User)
        and user.metadata.get("provider") == "credentials"
    ):
        # Handle password auth logout
        logger.info("Logging out password-authenticated user: %s", user.identifier)
        cl.user_session.set("user", None)
        cl.user_session.set("authenticated", False)
        return

    # Handle Supabase authentication logout
    if supabase_client:
        try:
            # Auth-py API expects sign_out to be called directly
            supabase_client.auth.sign_out()
            logger.info("User signed out from Supabase auth")
        except AuthApiError as e:
            logger.warning("Auth API error during Supabase sign out: %s", str(e))
        except AuthUnknownError as e:
            logger.warning("Auth unknown error during Supabase sign out: %s", str(e))
        except Exception as e:
            logger.warning("Exception during Supabase sign out: %s", str(e))

    # Clear local Chainlit session data
    cl.user_session.set("authenticated", False)
    cl.user_session.set("user", None)
    cl.user_session.set("supabase_session_storage", None)  # Clear stored session

    await cl.Message(
        content="👋 You have been successfully logged out.",
        author="System",
    ).send()
