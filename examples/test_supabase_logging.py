#!/usr/bin/env python3
"""
Simple test to verify Supabase logging is working correctly.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config import logger
from utils.supabase_logger import (
    log_request_to_supabase,
    setup_litellm_callbacks,
    update_user_token_usage,
)


def test_direct_logging():
    """Test direct logging to Supabase."""
    logger.info("Testing direct Supabase logging...")

    try:
        log_request_to_supabase(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test message"}],
            response={"content": "Test response", "usage": {"total_tokens": 50}},
            end_user="test_user",
            status="success",
            response_time=1.5,
            total_cost=0.001,
            user_profile_id="test_user_123",
            tokens_used=50,
            provider="openai",
        )
        logger.info("✅ Direct logging successful")
        return True
    except Exception as e:
        logger.error("❌ Direct logging failed: %s", str(e))
        return False


def test_token_usage_update():
    """Test token usage tracking."""
    logger.info("Testing token usage update...")

    try:
        update_user_token_usage(
            user_profile_id="test_user_123",
            tokens_used=25,
            cost=0.0005,
            model="gpt-4o-mini",
            provider="openai",
        )
        logger.info("✅ Token usage update successful")
        return True
    except Exception as e:
        logger.error("❌ Token usage update failed: %s", str(e))
        return False


def test_callback_setup():
    """Test LiteLLM callback setup."""
    logger.info("Testing LiteLLM callback setup...")

    try:
        setup_litellm_callbacks()
        logger.info("✅ Callback setup successful")
        return True
    except Exception as e:
        logger.error("❌ Callback setup failed: %s", str(e))
        return False


def main():
    """Run all tests."""
    logger.info("Starting Supabase logging tests...")
    logger.info("=" * 50)

    results = []

    # Test direct logging
    results.append(test_direct_logging())

    # Test token usage update
    results.append(test_token_usage_update())

    # Test callback setup
    results.append(test_callback_setup())

    logger.info("=" * 50)
    passed = sum(results)
    total = len(results)

    if passed == total:
        logger.info("✅ All tests passed (%d/%d)", passed, total)
    else:
        logger.warning("⚠️  Some tests failed (%d/%d)", passed, total)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
