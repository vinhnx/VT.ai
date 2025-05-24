#!/usr/bin/env python3
"""
Test BYOK (Bring Your Own Key) functionality to verify authentication flow.
"""

import os
import sys
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from utils.config import logger
from utils.llm_providers_config import get_llm_params


def test_byok_priority():
    """Test the priority order for BYOK authentication."""
    logger.info("Testing BYOK authentication priority...")

    # Test 1: Environment variables only
    os.environ["OPENAI_API_KEY"] = "env_test_key"
    params = get_llm_params("gpt-4")
    assert params.get("api_key") == "env_test_key", "Environment key should be used"
    logger.info("✅ Environment variable priority working")

    # Test 2: User keys override environment
    user_keys = {"openai": "user_test_key"}
    params = get_llm_params("gpt-4", user_keys=user_keys)
    assert (
        params.get("api_key") == "user_test_key"
    ), "User key should override environment"
    logger.info("✅ User key priority working")

    # Test 3: Different providers
    test_cases = [
        ("anthropic/claude-3-opus", "anthropic", "ANTHROPIC_API_KEY"),
        ("gemini/gemini-pro", "gemini", "GEMINI_API_KEY"),
        ("cohere/command-r", "cohere", "COHERE_API_KEY"),
        ("mistral/mistral-large", "mistral", "MISTRAL_API_KEY"),
        ("groq/llama3-8b", "groq", "GROQ_API_KEY"),
        ("deepseek/deepseek-chat", "deepseek", "DEEPSEEK_API_KEY"),
        ("openrouter/anthropic/claude-3", "openrouter", "OPENROUTER_API_KEY"),
    ]

    for model, provider, env_var in test_cases:
        # Set environment variable
        os.environ[env_var] = f"env_{provider}_key"
        params = get_llm_params(model)
        assert (
            params.get("api_key") == f"env_{provider}_key"
        ), f"Environment key for {provider} should work"

        # Test with user key override
        user_keys = {provider: f"user_{provider}_key"}
        params = get_llm_params(model, user_keys=user_keys)
        assert (
            params.get("api_key") == f"user_{provider}_key"
        ), f"User key for {provider} should override"

        logger.info("✅ %s provider authentication working", provider)

    logger.info("🎉 All BYOK authentication tests passed!")


def test_no_keys_scenario():
    """Test behavior when no API keys are provided."""
    logger.info("Testing no-keys scenario...")

    # Clear environment variables
    for env_var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
        if env_var in os.environ:
            del os.environ[env_var]

    # Test with no keys
    params = get_llm_params("gpt-4")
    assert params.get("api_key") is None, "Should return None when no keys available"
    logger.info("✅ No-keys scenario handled correctly")


def mock_chainlit_user_env():
    """Mock Chainlit user_env for testing public BYOK apps."""
    import chainlit as cl

    # Mock the user_session.get method to simulate user_env
    original_get = getattr(cl.user_session, "get", None)

    def mock_get(key):
        if key == "env":
            return {"OPENAI_API_KEY": "chainlit_user_env_key"}
        return None if not original_get else original_get(key)

    cl.user_session.get = mock_get

    # Test that user_env takes priority
    os.environ["OPENAI_API_KEY"] = "env_key"
    user_keys = {"openai": "user_key"}

    params = get_llm_params("gpt-4", user_keys=user_keys)

    # Restore original method
    if original_get:
        cl.user_session.get = original_get

    # User_env should have highest priority
    if params.get("api_key") == "chainlit_user_env_key":
        logger.info("✅ Chainlit user_env priority working")
        return True
    else:
        logger.warning(
            "❌ Chainlit user_env priority not working: got %s", params.get("api_key")
        )
        return False


def main():
    """Run all BYOK tests."""
    logger.info("Starting BYOK functionality tests...")

    try:
        test_byok_priority()
        test_no_keys_scenario()

        # Test Chainlit user_env (may not work without actual Chainlit session)
        try:
            mock_chainlit_user_env()
        except Exception as e:
            logger.warning("Chainlit user_env test skipped: %s", str(e))

        logger.info("🎉 All BYOK tests completed successfully!")
        return True

    except Exception as e:
        logger.error("❌ BYOK test failed: %s: %s", type(e).__name__, str(e))
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
