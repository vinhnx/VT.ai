"""
Utility for logging LLM usage to Supabase.
"""

import json
import time
from typing import Optional

import chainlit as cl
from supabase import Client

# Attempt to import supabase_client from app.py
# This creates a dependency, consider passing client as an argument for better decoupling if needed.
try:
    from vtai.app import supabase_client as global_supabase_client
except ImportError:
    global_supabase_client = None

from vtai.utils.config import logger

# Supabase client will be imported from app.py or passed as an argument
# For now, we assume it's available globally or passed to functions.
# This will be refined once app structure for global objects is clearer.


async def log_usage_to_supabase(
    user_id: Optional[str],
    session_id: str,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cost: Optional[float] = None,
) -> None:
    """
    Logs LLM usage data to the Supabase 'request_logs' table.

    Args:
        user_id: The ID of the authenticated user (if available).
        session_id: The current Chainlit session ID.
        model_name: The name of the LLM used.
        input_tokens: The number of input tokens.
        output_tokens: The number of output tokens.
        total_tokens: The total number of tokens.
        cost: The calculated cost of the API call (optional).
    """
    # Use the globally imported client, or allow override if passed (though not in signature now)
    client_to_use = global_supabase_client
    if not client_to_use:
        logger.warning(
            "Supabase client not available globally. Skipping usage logging."
        )
        return

    # Debug logging
    logger.debug(
        f"Logging usage to request_logs table - Model: {model_name}, User: {user_id or 'anonymous'}, Session: {session_id}"
    )
    logger.debug(
        f"Token counts - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}"
    )

    # Check if user is authenticated with built-in password system rather than Supabase
    user = cl.user_session.get("user")
    if (
        user
        and hasattr(user, "metadata")
        and user.metadata.get("provider") == "credentials"
    ):
        logger.info("User authenticated via password auth. Skipping usage logging.")
        return

    try:
        # Format the data for the request_logs table structure
        request_log_entry = {
            "model": model_name,
            "messages": "[]",  # Empty JSON array as string
            "response": "{}",  # Empty JSON object as string
            "end_user": user_id or session_id,
            "status": "success",
            "response_time": 0.0,  # Not available from this function
            "total_cost": cost,
            "additional_details": json.dumps(
                {
                    "session_id": session_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "timestamp": time.time(),
                    "migrated_from_usage_logger": True,
                }
            ),
            "litellm_call_id": f"legacy-{int(time.time())}-{session_id[-8:]}",
        }

        # Create the query without awaiting it
        query = client_to_use.table("request_logs").insert(request_log_entry)

        # Execute the query - handle as non-awaitable
        try:
            result = query.execute()

            if hasattr(result, "data") and result.data:
                logger.info(
                    f"Successfully logged usage for model {model_name} to request_logs table."
                )
            else:
                logger.warning(
                    f"Usage logging to request_logs table for model {model_name} might have failed. "
                    f"Error: {getattr(result, 'error', 'Unknown error')}"
                )
        except Exception as query_error:
            logger.error(f"Error executing Supabase query: {query_error}")

    except Exception as e:
        # Handle RLS policy violations more gracefully
        if "violates row-level security policy" in str(e):
            logger.warning(
                f"Row-level security policy prevented logging usage to request_logs table. "
                f"This may happen if the user doesn't have the right permissions."
            )
        else:
            logger.error(f"Error logging usage to request_logs table: {e}")


def get_user_id_from_session() -> Optional[str]:
    """
    Retrieves the user ID from the Chainlit user session.
    """
    try:
        # Check if the user is authenticated
        is_authenticated = cl.user_session.get("authenticated", False)

        if not is_authenticated:
            logger.debug("User not authenticated, returning None")
            return None

        user_info = cl.user_session.get("user")

        # Handle different user object structures
        if user_info:
            # Handle dict structure (from Supabase auth)
            if isinstance(user_info, dict):
                user_id = user_info.get("id")
                logger.debug(f"Retrieved user ID from dict: {user_id}")
                return user_id
            # Handle cl.User object structure (from password auth)
            elif hasattr(user_info, "identifier"):
                user_id = user_info.identifier
                logger.debug(f"Retrieved user ID from User object: {user_id}")
                return user_id
            # Log if we don't understand the user_info structure
            else:
                logger.warning(f"Unknown user_info structure: {type(user_info)}")
                logger.debug(f"User info content: {user_info}")

        logger.debug("No user ID found in session")
        return None
    except Exception as e:
        logger.error(f"Error retrieving user ID from session: {e}")
        return None


def get_session_id() -> str:
    """
    Retrieves the current Chainlit session ID.
    """
    try:
        session_id = cl.user_session.id()
        if not session_id:
            logger.warning("Chainlit session ID is empty, using fallback")
            session_id = f"fallback-{int(time.time())}"
        logger.debug(f"Retrieved session ID: {session_id}")
        return session_id
    except Exception as e:
        logger.error(f"Error retrieving session ID: {e}")
        return f"error-{int(time.time())}"
