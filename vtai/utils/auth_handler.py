"""
Authentication handling for VT.ai.

Manages both Supabase and simple password-based authentication, including login,
OAuth flows, session management, password reset, and email verification. Provides
comprehensive integration with Supabase for user profile synchronization.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import chainlit as cl
import dotenv
from gotrue.errors import AuthApiError, AuthUnknownError
from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions

from vtai.utils.config import logger

# Load environment variables
dotenv.load_dotenv()

# First attempt to import supabase_client from vtai.app
# If that fails, initialize it locally
try:
    from vtai.app import supabase_client

    logger.info("Supabase client imported from vtai.app")
except ImportError:
    # Initialize Supabase client locally
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
            supabase_client: Optional[Client] = create_client(
                SUPABASE_URL, SUPABASE_KEY
            )
            logger.info("Supabase client initialized locally in auth_handler.py")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client locally: {e}")
            supabase_client = None

from datetime import datetime  # Added for timestamping


@cl.oauth_callback
async def oauth_callback(
    provider_id: str, token: str, raw_user_data: Dict[str, Any], default_user: cl.User
) -> Optional[cl.User]:
    """
    OAuth callback for Chainlit's OAuth authentication system.

    This function is called when a user successfully authenticates via OAuth.
    It enhances the default user object with information from the provider
    and syncs user data with the Supabase 'user_profiles' table.

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
    logger.info(f"Token: {token[:15]}... (truncated for security)")
    logger.info(f"Raw User Data from {provider_id}: {raw_user_data}")
    logger.info(
        f"Default User (incoming): Identifier: {default_user.identifier}, Metadata: {default_user.metadata}"
    )
    logger.info("-----------------------------------------------------")

    email: Optional[str] = None
    name: Optional[str] = None
    picture: Optional[str] = None
    user_provider_specific_id: Optional[str] = None

    if provider_id == "google":
        # Google can return either "sub" or "id" field for the user ID
        user_provider_specific_id = raw_user_data.get("sub") or raw_user_data.get("id")
        if user_provider_specific_id is None:
            # As a last resort, if both are missing, try to use a hashed email as ID
            email = raw_user_data.get("email")
            if email:
                import hashlib

                user_provider_specific_id = hashlib.md5(email.encode()).hexdigest()
                logger.warning(
                    f"Using MD5 hash of email as provider_user_id: {user_provider_specific_id}"
                )

        email = raw_user_data.get("email")
        name = raw_user_data.get("name")
        picture = raw_user_data.get("picture")
        logger.info(
            f"Google OAuth details extracted - user_id: {user_provider_specific_id}, email: {email}, name: {name}"
        )
    elif provider_id == "github":
        gh_id = raw_user_data.get("id")
        user_provider_specific_id = str(gh_id) if gh_id is not None else None
        email = raw_user_data.get("email")  # May be null
        name = raw_user_data.get("name") or raw_user_data.get("login")
        picture = raw_user_data.get("avatar_url")
        logger.info(
            f"GitHub OAuth details extracted - id: {user_provider_specific_id}, email: {email}, name: {name}"
        )
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
        enhanced_user.metadata["name"] = name
    if picture:
        enhanced_user.metadata["picture"] = picture
    if user_provider_specific_id:
        enhanced_user.metadata["user_provider_id"] = user_provider_specific_id

    logger.info(
        f"Enhanced User prepared: Identifier: {enhanced_user.identifier}, Metadata: {enhanced_user.metadata}"
    )

    # Log supabase_client availability
    logger.info(f"Supabase client available: {supabase_client is not None}")

    # Sync with Supabase user_profiles table
    if supabase_client:
        profile_to_sync = {
            "provider": enhanced_user.metadata.get("provider"),
            "provider_user_id": enhanced_user.metadata.get("user_provider_id"),
            "email": enhanced_user.metadata.get("email"),
            "full_name": enhanced_user.metadata.get("name"),
            "avatar_url": enhanced_user.metadata.get("picture"),
            "chainlit_user_identifier": enhanced_user.identifier,
            "raw_oauth_details": json.dumps(raw_user_data),  # Ensure JSON serialized
            "updated_at": datetime.utcnow().isoformat()
            + "+00:00",  # ISO 8601 format for Supabase
        }

        # Log the profile before filtering
        logger.info(
            f"Profile to sync (before filtering None values): {profile_to_sync}"
        )

        # Filter out None values, but use empty string for provider_user_id if it's None
        # This helps prevent issues with Supabase constraints
        profile_to_sync = {
            k: (v if v is not None or k != "provider_user_id" else "")
            for k, v in profile_to_sync.items()
            if v is not None or k == "provider_user_id"
        }

        # Extra safety check: If provider_user_id is still None or empty after filtering,
        # generate a random ID as a last resort
        if not profile_to_sync.get("provider_user_id"):
            import uuid

            profile_to_sync["provider_user_id"] = str(uuid.uuid4())
            logger.warning(
                f"Generated random UUID as provider_user_id: {profile_to_sync['provider_user_id']}"
            )

        # Log the profile after filtering
        logger.info(f"Profile to sync (after filtering None values): {profile_to_sync}")

        if profile_to_sync.get("provider") and profile_to_sync.get("provider_user_id"):
            try:
                logger.info(
                    f"Attempting to sync profile for provider '{profile_to_sync['provider']}' "
                    f"and user_id '{profile_to_sync['provider_user_id']}' with Supabase."
                )

                select_response = None
                try:
                    # Try to check if the user exists
                    select_response = (
                        await supabase_client.table("user_profiles")
                        .select("id")
                        .eq("provider", profile_to_sync["provider"])
                        .eq("provider_user_id", profile_to_sync["provider_user_id"])
                        .maybe_single()
                        .execute()
                    )
                except Exception as select_e:
                    logger.error(
                        f"Error during select query: {type(select_e).__name__} - {select_e}"
                    )
                    # Continue with insert even if select fails

                # Log the select response
                logger.info(f"Supabase select response: {select_response}")
                logger.info(
                    f"Supabase select data: {getattr(select_response, 'data', None)}"
                )
                logger.info(
                    f"Supabase select error: {getattr(select_response, 'error', None)}"
                )

                select_data = getattr(select_response, "data", None)
                select_error = getattr(select_response, "error", None)

                if select_error:
                    logger.error(
                        f"Supabase select error: {getattr(select_error, 'message', select_error)}"
                    )
                    # Try to insert anyway since select failed
                    logger.info("Attempting insert after select error...")
                    insert_payload = profile_to_sync.copy()
                    now_iso = datetime.utcnow().isoformat() + "+00:00"
                    insert_payload["created_at"] = now_iso
                    insert_payload["updated_at"] = now_iso

                    try:
                        insert_response = (
                            await supabase_client.table("user_profiles")
                            .insert(insert_payload)
                            .execute()
                        )

                        # Log the insert response
                        logger.info(f"Supabase insert response: {insert_response}")
                        logger.info(
                            f"Supabase insert data: {getattr(insert_response, 'data', None)}"
                        )
                        logger.info(
                            f"Supabase insert error: {getattr(insert_response, 'error', None)}"
                        )
                    except Exception as insert_e:
                        logger.error(
                            f"Error during insert after select error: {type(insert_e).__name__} - {insert_e}"
                        )
                elif select_data:  # User exists, update them
                    db_user_id = select_response.data["id"]
                    logger.info(
                        f"Updating existing user profile (ID: {db_user_id}) in Supabase."
                    )
                    update_payload = profile_to_sync.copy()
                    update_payload["updated_at"] = (
                        datetime.utcnow().isoformat() + "+00:00"
                    )

                    update_response = (
                        await supabase_client.table("user_profiles")
                        .update(update_payload)
                        .eq("id", db_user_id)
                        .execute()
                    )

                    # Log the update response
                    logger.info(f"Supabase update response: {update_response}")
                    logger.info(
                        f"Supabase update data: {getattr(update_response, 'data', None)}"
                    )
                    logger.info(
                        f"Supabase update error: {getattr(update_response, 'error', None)}"
                    )

                    update_error = getattr(update_response, "error", None)
                    if update_error:
                        logger.error(
                            f"Supabase update error: {getattr(update_error, 'message', update_error)}"
                        )
                    else:
                        logger.info("User profile updated successfully in Supabase.")
                else:  # User does not exist, use upsert to handle both insertion and conflict cases
                    logger.info("Using upsert for user profile in Supabase.")
                    upsert_payload = profile_to_sync.copy()
                    now_iso = datetime.utcnow().isoformat() + "+00:00"
                    upsert_payload["created_at"] = now_iso
                    upsert_payload["updated_at"] = now_iso

                    # Use upsert with on_conflict parameter to handle duplicates
                    # This will insert if the record doesn't exist, and do nothing if it does
                    upsert_response = (
                        await supabase_client.table("user_profiles")
                        .upsert(
                            upsert_payload,
                            on_conflict="provider,provider_user_id",
                            ignoreDuplicates=True,  # This ensures existing records are not updated
                        )
                        .execute()
                    )

                    # Log the upsert response
                    logger.info(f"Supabase upsert response: {upsert_response}")
                    logger.info(
                        f"Supabase upsert data: {getattr(upsert_response, 'data', None)}"
                    )
                    logger.info(
                        f"Supabase upsert error: {getattr(upsert_response, 'error', None)}"
                    )

                    upsert_error = getattr(upsert_response, "error", None)
                    if upsert_error:
                        logger.error(
                            f"Supabase upsert error: {getattr(upsert_error, 'message', upsert_error)}"
                        )
                    else:
                        logger.info(
                            "User profile upsert completed successfully in Supabase."
                        )

            except Exception as e:
                logger.error(
                    f"Error during Supabase profile sync: {type(e).__name__} - {e}"
                )
                logger.exception("Traceback for Supabase profile sync error:")

                # Final fallback - try a direct upsert with explicit error handling
                try:
                    logger.info("Attempting direct upsert as final fallback...")
                    fallback_payload = profile_to_sync.copy()
                    now_iso = datetime.utcnow().isoformat() + "+00:00"
                    fallback_payload["created_at"] = now_iso
                    fallback_payload["updated_at"] = now_iso

                    upsert_response = (
                        await supabase_client.table("user_profiles")
                        .upsert(
                            fallback_payload,
                            on_conflict="provider,provider_user_id",
                            ignoreDuplicates=True,  # This ensures existing records are not updated
                        )
                        .execute()
                    )

                    logger.info(f"Fallback upsert response: {upsert_response}")
                    logger.info(
                        f"Fallback upsert data: {getattr(upsert_response, 'data', None)}"
                    )
                except Exception as fallback_e:
                    logger.error(
                        f"Fallback upsert also failed: {type(fallback_e).__name__} - {fallback_e}"
                    )
        else:
            logger.warning(
                "Provider or provider_user_id missing in enhanced_user.metadata. "
                "Skipping Supabase profile sync."
            )
    else:
        logger.info("Supabase client not available. Skipping user profile sync.")

    # We won't try to set user session variables here because of Chainlit context issues
    # The ChainlitContextException occurs because OAuth callback runs outside Chainlit's context
    # Instead, Chainlit will automatically use the returned enhanced_user to set up the user session
    logger.info(
        "Skipping manual user session variable setting due to potential Chainlit context issues"
    )
    logger.info(
        "Chainlit will set up the user session based on the enhanced_user we return"
    )

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
        cl.user_session.set("authenticated", False)
        cl.user_session.set("user", None)
        return

    cl.user_session.set("authenticated", False)
    cl.user_session.set("user", None)

    stored_session_data = cl.user_session.get("supabase_session_storage")

    if stored_session_data:
        try:
            user_id = stored_session_data.get("user", {}).get("id")
            logger.debug(
                "Attempting to restore session for user if ID is present: %s", user_id
            )
            supabase_client.auth.set_session(
                access_token=stored_session_data["access_token"],
                refresh_token=stored_session_data["refresh_token"],
            )
            response = supabase_client.auth.get_user()

            if response and response.user:
                user_data = response.user.model_dump()
                current_session_details = supabase_client.auth.get_session()
                if not current_session_details:
                    logger.warning(
                        "Failed to get current session details after get_user succeeded. Clearing stored session."
                    )
                    cl.user_session.set("supabase_session_storage", None)
                    return

                cl.user_session.set("user", user_data)
                cl.user_session.set("authenticated", True)
                cl.user_session.set(
                    "supabase_session_storage", current_session_details.model_dump()
                )

                logger.info(
                    "User %s authenticated from stored session.", user_data.get("id")
                )
                return
            else:
                logger.warning(
                    "Stored session found but get_user() failed. Clearing stored session."
                )
                cl.user_session.set("supabase_session_storage", None)

        except AuthApiError as e:
            logger.warning(
                "Failed to validate stored session due to auth API error: %s. Clearing stored session.",
                str(e),
            )
            cl.user_session.set("supabase_session_storage", None)
            try:
                supabase_client.auth.sign_out()
            except Exception:
                pass

        except AuthUnknownError as e:
            logger.warning(
                "Failed to validate stored session due to auth unknown error: %s. Clearing stored session.",
                str(e),
            )
            cl.user_session.set("supabase_session_storage", None)
            try:
                supabase_client.auth.sign_out()
            except Exception:
                pass

        except Exception as e:
            logger.warning(
                "Failed to validate stored session: %s. Clearing stored session.",
                str(e),
            )
            cl.user_session.set("supabase_session_storage", None)
            try:
                supabase_client.auth.sign_out()
            except Exception:
                pass

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
        response = supabase_client.auth.sign_up(
            {
                "email": email,
                "password": password,
                "options": {
                    "data": user_metadata or {},
                },
            }
        )

        if response.user and response.session:
            user_data = response.user.model_dump()
            session_data = response.session.model_dump()
            cl.user_session.set("user", user_data)
            cl.user_session.set("authenticated", True)
            cl.user_session.set("supabase_session_storage", session_data)
            supabase_client.auth.set_session(
                session_data["access_token"], session_data["refresh_token"]
            )
            logger.info(
                "User %s signed up and logged in successfully.", user_data.get("id")
            )
            return user_data, None
        elif response.user and not response.session:
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
        response = supabase_client.auth.sign_in_with_password(
            {"email": email, "password": password}
        )

        if response.user and response.session:
            user_data = response.user.model_dump()
            session_data = response.session.model_dump()
            cl.user_session.set("user", user_data)
            cl.user_session.set("authenticated", True)
            cl.user_session.set("supabase_session_storage", session_data)
            supabase_client.auth.set_session(
                session_data["access_token"], session_data["refresh_token"]
            )
            logger.info("User %s logged in successfully.", user_data.get("id"))
            return user_data, None
        else:
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
        response = supabase_client.auth.sign_in_with_oauth(
            {
                "provider": provider,
                "options": {
                    "redirect_to": redirect_to,
                },
            }
        )

        logger.info("Initiated OAuth flow for provider: %s", provider)

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
    return "basic_test"


async def check_authentication(
    supabase_client: Optional[Client],
) -> Tuple[bool, Optional[Any]]:
    """
    Checks if the current user is authenticated based on Chainlit session state.
    Supports both Supabase authentication and password-based authentication.

    Args:
            supabase_client: The initialized Supabase client (currently unused here but kept for signature consistency).

    Returns:
            A tuple containing (is_authenticated, user_data)
    """
    user = cl.user_session.get("user")
    if (
        user
        and isinstance(user, cl.User)
        and user.metadata.get("provider") == "credentials"
    ):
        return True, user

    is_authenticated = cl.user_session.get("authenticated", False)
    user_data = cl.user_session.get("user")

    if is_authenticated and user_data:
        return True, user_data

    if not is_authenticated:
        cl.user_session.set("user", None)
        cl.user_session.set("supabase_session_storage", None)

    return False, None


async def logout(supabase_client: Optional[Client]) -> None:
    """
    Logs out the current user.
    Supports both Supabase and password-based authentication.

    Args:
        supabase_client: The initialized Supabase client
    """
    user = cl.user_session.get("user")
    if (
        user
        and isinstance(user, cl.User)
        and user.metadata.get("provider") == "credentials"
    ):
        logger.info("Logging out password-authenticated user: %s", user.identifier)
        cl.user_session.set("user", None)
        cl.user_session.set("authenticated", False)
        return

    if supabase_client:
        try:
            supabase_client.auth.sign_out()
            logger.info("User signed out from Supabase auth")
        except AuthApiError as e:
            logger.warning("Auth API error during Supabase sign out: %s", str(e))
        except AuthUnknownError as e:
            logger.warning("Auth unknown error during Supabase sign out: %s", str(e))
        except Exception as e:
            logger.warning("Exception during Supabase sign out: %s", str(e))

    cl.user_session.set("authenticated", False)
    cl.user_session.set("user", None)
    cl.user_session.set("supabase_session_storage", None)

    await cl.Message(
        content="👋 You have been successfully logged out.",
        author="System",
    ).send()
