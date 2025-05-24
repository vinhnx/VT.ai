#!/usr/bin/env python3
"""
Test user analytics and relationship functionality.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config import logger
from utils.supabase_logger import (
    get_recent_user_activity,
    get_user_analytics,
    get_user_request_history,
    get_user_token_breakdown,
)


def test_user_analytics():
    """Test user analytics functionality."""
    logger.info("Testing user analytics...")

    test_user_id = "test_user_123"

    # Get user analytics
    analytics = get_user_analytics(test_user_id)
    if analytics:
        logger.info("✅ User Analytics for %s:", test_user_id)
        logger.info("  Email: %s", analytics.get("email"))
        logger.info("  Total Requests: %s", analytics.get("total_requests"))
        logger.info("  Successful Requests: %s", analytics.get("successful_requests"))
        logger.info("  Failed Requests: %s", analytics.get("failed_requests"))
        logger.info("  Total Tokens: %s", analytics.get("total_tokens_from_logs"))
        logger.info("  Total Cost: $%s", analytics.get("total_cost"))
        logger.info("  Most Used Model: %s", analytics.get("most_used_model"))
        logger.info("  Most Used Provider: %s", analytics.get("most_used_provider"))
        logger.info("  Last Request: %s", analytics.get("last_request_time"))
    else:
        logger.warning("❌ No analytics found for user %s", test_user_id)

    return analytics is not None


def test_request_history():
    """Test user request history."""
    logger.info("Testing user request history...")

    test_user_id = "test_user_123"

    # Get request history
    history = get_user_request_history(test_user_id, limit=5)
    if history:
        logger.info(
            "✅ Request History for %s (%d requests):", test_user_id, len(history)
        )
        for i, request in enumerate(history[:3], 1):  # Show first 3
            logger.info(
                "  %d. Model: %s, Status: %s, Tokens: %s, Time: %s",
                i,
                request.get("model"),
                request.get("status"),
                request.get("tokens_used"),
                request.get("request_time"),
            )
    else:
        logger.warning("❌ No request history found for user %s", test_user_id)

    return len(history) > 0 if history else False


def test_token_breakdown():
    """Test user token breakdown."""
    logger.info("Testing user token breakdown...")

    test_user_id = "test_user_123"

    # Get token breakdown
    breakdown = get_user_token_breakdown(test_user_id)
    if breakdown:
        logger.info("✅ Token Breakdown for %s:", test_user_id)
        for item in breakdown:
            logger.info(
                "  Model: %s (%s) - Requests: %s, Tokens: %s, Cost: $%s",
                item.get("model"),
                item.get("provider"),
                item.get("request_count"),
                item.get("total_tokens"),
                item.get("total_cost"),
            )
    else:
        logger.warning("❌ No token breakdown found for user %s", test_user_id)

    return len(breakdown) > 0 if breakdown else False


def test_recent_activity():
    """Test recent activity across all users."""
    logger.info("Testing recent activity...")

    # Get recent activity
    activity = get_recent_user_activity(limit=5)
    if activity:
        logger.info("✅ Recent Activity (%d requests):", len(activity))
        for i, request in enumerate(activity[:3], 1):  # Show first 3
            logger.info(
                "  %d. User: %s, Model: %s, Status: %s, Time: %s",
                i,
                request.get("email") or request.get("user_id") or "Anonymous",
                request.get("model"),
                request.get("status"),
                request.get("request_time"),
            )
    else:
        logger.warning("❌ No recent activity found")

    return len(activity) > 0 if activity else False


def main():
    """Test all analytics functionality."""
    logger.info("Testing User Analytics & Relationship Functionality")
    logger.info("=" * 60)

    results = []

    # Test all functions
    results.append(test_user_analytics())
    logger.info("-" * 40)

    results.append(test_request_history())
    logger.info("-" * 40)

    results.append(test_token_breakdown())
    logger.info("-" * 40)

    results.append(test_recent_activity())

    logger.info("=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        logger.info("✅ All analytics tests passed (%d/%d)", passed, total)
        logger.info("🎉 User-Request relationship is working perfectly!")
    else:
        logger.warning("⚠️  Some analytics tests failed (%d/%d)", passed, total)

    return passed == total


if __name__ == "__main__":
    main()
