#!/usr/bin/env python3
"""
Test LiteLLM callbacks with actual completion calls.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import litellm
from utils.config import logger
from utils.supabase_logger import setup_litellm_callbacks


def test_litellm_with_callbacks():
    """Test LiteLLM completion with custom callbacks."""
    logger.info("Testing LiteLLM with custom Supabase callbacks...")

    # Setup callbacks
    setup_litellm_callbacks()

    try:
        # Test with a simple model that should work (if available)
        # Using ollama as it's likely to be available locally
        response = litellm.completion(
            model="ollama/llama3.2:1b",
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Hello from LiteLLM' in exactly those words",
                }
            ],
            user="test_callback_user_456",
            max_tokens=10,
            temperature=0,
        )

        if response:
            logger.info("✅ LiteLLM completion successful")
            logger.info("Response: %s", response.choices[0].message.content)
            logger.info(
                "Usage: %s",
                response.usage if hasattr(response, "usage") else "No usage data",
            )
            return True
        else:
            logger.warning("⚠️  No response received")
            return False

    except Exception as e:
        logger.error("❌ LiteLLM completion failed: %s", str(e))
        return False


def test_error_callback():
    """Test the error callback with an invalid model."""
    logger.info("Testing error callback...")

    try:
        # This should trigger the failure callback
        response = litellm.completion(
            model="invalid/nonexistent-model",
            messages=[{"role": "user", "content": "This should fail"}],
            user="test_error_user_789",
        )

    except Exception as e:
        logger.info("✅ Expected error occurred: %s", str(e)[:100])
        return True

    logger.warning("⚠️  Expected error did not occur")
    return False


def main():
    """Test callback integration."""
    logger.info("Testing LiteLLM callback integration")
    logger.info("=" * 60)

    results = []

    # Test successful completion
    results.append(test_litellm_with_callbacks())

    # Test error handling
    results.append(test_error_callback())

    logger.info("=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        logger.info("✅ All callback tests passed (%d/%d)", passed, total)
        logger.info("Check your Supabase request_logs table for logged requests")
    else:
        logger.warning("⚠️  Some callback tests failed (%d/%d)", passed, total)

    return passed == total


if __name__ == "__main__":
    main()
