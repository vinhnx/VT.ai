"""
User credits management for VT.ai

Implements daily credit system using Supabase request_logs.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from vtai.utils.config import logger
from vtai.utils.supabase_client import (
    get_user_analytics,
    get_user_successful_requests_today,
)

# Default daily credits for free tier
DEFAULT_DAILY_CREDITS = 100


def get_credits_reset_time() -> datetime:
    """Get the next daily reset time (23:59 UTC today)."""
    now = datetime.now(timezone.utc)
    reset = now.replace(hour=23, minute=59, second=0, microsecond=0)
    if now > reset:
        reset = reset + timedelta(days=1)
    return reset


def get_user_credits(user_id: str) -> dict:
    """Return current credits, max credits, and reset time for a user."""
    try:
        today_success = get_user_successful_requests_today(user_id)
        logger.info("[CREDITS] user_id=%s today_success=%s", user_id, today_success)
        credits_left = max(DEFAULT_DAILY_CREDITS - today_success, 0)
        logger.info(
            "[CREDITS] Calculated credits_left=%s for user %s", credits_left, user_id
        )
        return {
            "credits_left": credits_left,
            "max_credits": DEFAULT_DAILY_CREDITS,
            "reset_time": get_credits_reset_time().isoformat(),
        }
    except Exception as e:
        logger.error("Error: %s: %s", type(e).__name__, str(e))
        return {
            "credits_left": DEFAULT_DAILY_CREDITS,
            "max_credits": DEFAULT_DAILY_CREDITS,
            "reset_time": get_credits_reset_time().isoformat(),
        }


def get_user_credits_info(user_id: str) -> dict:
    """Return user credits info for UI (current/total)."""
    return get_user_credits(user_id)


def check_user_can_chat(user_id: str) -> bool:
    """Return True if user has credits left, else False."""
    info = get_user_credits(user_id)
    return info["credits_left"] > 0
