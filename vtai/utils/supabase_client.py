"""
Supabase client utilities and user profile helpers for VT.ai.
"""

import os
from typing import Optional

from chainlit import CustomElement, Message, run_sync, user_session
from supabase import Client as SupabaseClient
from supabase import create_client

from vtai.utils.config import logger

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

supabase_client: Optional[SupabaseClient] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    logger.warning("Supabase credentials not set. Logging will be disabled.")


class UserProfileService:
    """Service for user profile operations with Supabase and Chainlit UI."""

    @staticmethod
    def fetch_user_profile_from_supabase(user_id: str) -> dict:
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
            logger.error(
                "Error fetching user profile: %s: %s", type(e).__name__, str(e)
            )
            return {}

    @staticmethod
    def show_user_profile_action() -> None:
        from vtai.utils.user_session_helper import get_user_profile

        user_id = user_session.get("user_id")
        profile = get_user_profile() or {}
        if not profile and user_id:
            profile = UserProfileService.fetch_user_profile_from_supabase(user_id)
        if not profile:
            run_sync(Message(content="No user profile found.").send())
        else:
            run_sync(
                Message(
                    content="Your profile:",
                    elements=[CustomElement(name="UserProfile", props=profile)],
                ).send()
            )

    @staticmethod
    def show_user_profile_action() -> None:
        from vtai.utils.user_session_helper import get_user_profile

        user_id = user_session.get("user_id")
        profile = get_user_profile() or {}
        if not profile and user_id:
            profile = UserProfileService.fetch_user_profile_from_supabase(user_id)
        if not profile:
            run_sync(Message(content="No user profile found.").send())
        else:
            run_sync(
                Message(
                    content="Your profile:",
                    elements=[CustomElement(name="UserProfile", props=profile)],
                ).send()
            )


def log_request_to_supabase(*_args, **_kwargs) -> None:
    """Stub for logging requests to Supabase (no-op)."""
    return


def setup_litellm_callbacks() -> None:
    """Stub for setting up LiteLLM callbacks (no-op)."""
    return
    return
