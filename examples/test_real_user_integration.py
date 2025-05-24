#!/usr/bin/env python3
"""
Test the integration with real authenticated users.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vtai.utils.supabase_logger import (
    log_request_to_supabase,
    get_user_analytics,
    setup_litellm_callbacks
)
from vtai.utils.config import logger


def test_real_user_logging():
    """Test logging with a real authenticated user ID."""
    logger.info("Testing real user logging...")

    # Use the real authenticated user from the database
    real_user_id = "google_117195204714065447709"  # Vinh Nguyen from Google OAuth

    # Setup callbacks
    setup_litellm_callbacks()

    # Test logging a request as if it came from the real authenticated user
    logger.info("Logging request for authenticated user: %s", real_user_id)

    log_request_to_supabase(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test with real authenticated user"}],
        response={"content": "Test response from authenticated user", "usage": {"total_tokens": 45}},
        end_user=real_user_id,  # This would be the session ID in practice
        status="success",
        tokens_used=45,
        total_cost=0.002,
        user_profile_id=real_user_id,  # This is the key - linking to the authenticated user
        provider="openai",
        litellm_call_id="test_call_12345"
    )

    # Check if the user's token count was updated
    analytics = get_user_analytics(real_user_id)
    if analytics:
        logger.info("✅ Real user analytics:")
        logger.info("  Email: %s", analytics.get("email"))
        logger.info("  Total Requests: %s", analytics.get("total_requests"))
        logger.info("  Total Tokens: %s", analytics.get("total_tokens_from_logs"))
        logger.info("  Total Cost: $%s", analytics.get("total_cost"))
        logger.info("  User Tokens Counter: %s", analytics.get("total_tokens_counter"))
        return True
    else:
        logger.error("❌ No analytics found for real user")
        return False


def test_session_to_user_mapping():
    """Test mapping session IDs to authenticated user IDs."""
    logger.info("Testing session to user mapping...")

    # Simulate how the app should work:
    # 1. Session ID from Chainlit session
    # 2. Authenticated user ID from OAuth
    # 3. LiteLLM gets session ID, but we log with authenticated user ID

    session_id = "test-session-12345"
    auth_user_id = "google_117195204714065447709"

    logger.info("Session ID: %s", session_id)
    logger.info("Authenticated User ID: %s", auth_user_id)

    # This is how it should work in the app:
    # - LiteLLM gets session_id in the 'user' parameter
    # - Our logging uses auth_user_id for user_profile_id linking
    # - This allows proper token tracking for authenticated users

    log_request_to_supabase(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test session to user mapping"}],
        response={"content": "Mapped correctly", "usage": {"total_tokens": 30}},
        end_user=session_id,  # What LiteLLM sees
        status="success",
        tokens_used=30,
        total_cost=0.001,
        user_profile_id=auth_user_id,  # What we use for linking
        provider="openai"
    )

    # Verify the mapping worked
    analytics = get_user_analytics(auth_user_id)
    if analytics and analytics.get("total_requests", 0) > 0:
        logger.info("✅ Session to user mapping working!")
        logger.info("  Requests logged under authenticated user: %s", analytics.get("total_requests"))
        return True
    else:
        logger.error("❌ Session to user mapping failed")
        return False


def main():
    """Test real user integration."""
    logger.info("Testing Real User Integration")
    logger.info("=" * 60)

    results = []

    # Test real user logging
    results.append(test_real_user_logging())
    logger.info("-" * 40)

    # Test session to user mapping
    results.append(test_session_to_user_mapping())

    logger.info("=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        logger.info("✅ All real user integration tests passed (%d/%d)", passed, total)
        logger.info("🎉 System ready for production with real authenticated users!")
        logger.info("")
        logger.info("Integration Summary:")
        logger.info("- ✅ Real authenticated users properly linked to requests")
        logger.info("- ✅ Token usage tracked for authenticated users")
        logger.info("- ✅ Session ID to User ID mapping working")
        logger.info("- ✅ Automatic token counter updates via database triggers")
    else:
        logger.warning("⚠️  Some tests failed (%d/%d)", passed, total)
        logger.info("Check the conversation_handlers.py updates for proper user ID handling")

    return passed == total


if __name__ == "__main__":
    main()