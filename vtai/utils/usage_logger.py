"""
Utility for logging LLM usage to Supabase.
"""

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
    Logs LLM usage data to the Supabase 'usage_logs' table.

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

    # Check if user is authenticated with built-in password system rather than Supabase
    user = cl.user_session.get("user")
    if (
        user
        and hasattr(user, "metadata")
        and user.metadata.get("provider") == "credentials"
    ):
        logger.info("User authenticated via password auth. Skipping usage logging.")
        return

    log_entry = {
        "user_id": user_id,
        "session_id": session_id,
        "model_name": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
        # timestamp is handled by Supabase (default now())
    }

    try:
        # Supabase table name is 'usage_logs'
        data, count = (
            await client_to_use.table("usage_logs").insert(log_entry).execute()
        )
        if (
            count is not None
            and len(count) > 0
            and hasattr(count[1], "__len__")
            and len(count[1]) > 0
        ):  # Check if insertion was successful based on returned data structure
            logger.info(
                f"Successfully logged usage for model {model_name} to Supabase."
            )
        else:
            if data and data[0]:
                logger.info(
                    f"Successfully logged usage for model {model_name} to Supabase. Data: {data[0]}"
                )
            else:
                logger.warning(
                    f"Usage logging to Supabase for model {model_name} might have failed or returned an unexpected response. Data: {data}, Count: {count}"
                )

    except Exception as e:
        # Handle RLS policy violations more gracefully
        if "violates row-level security policy" in str(e):
            logger.warning(
                f"Row-level security policy prevented logging usage to Supabase. "
                f"This may happen if the user doesn't have the right permissions."
            )
        else:
            logger.error(f"Error logging usage to Supabase: {e}")


def get_user_id_from_session() -> Optional[str]:
    """
    Retrieves the user ID from the Chainlit user session.
    """
    # Check if the user is authenticated
    is_authenticated = cl.user_session.get("authenticated", False)

    if not is_authenticated:
        return None

    user_info = cl.user_session.get("user")

    # Handle different user object structures
    if user_info:
        # Handle dict structure (from Supabase auth)
        if isinstance(user_info, dict):
            return user_info.get("id")
        # Handle cl.User object structure (from password auth)
        elif hasattr(user_info, "identifier"):
            return user_info.identifier

    return None


def get_session_id() -> str:
    """
    Retrieves the current Chainlit session ID.
    """
    return cl.user_session.id()  # Ensure this is the correct way to get session ID
