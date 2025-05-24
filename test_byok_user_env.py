#!/usr/bin/env python3
"""
Test script to verify BYOK user_env functionality works correctly.

This script tests the user_env authentication priority chain by:
1. Clearing environment variables to simulate missing API keys
2. Setting up a mock user_env scenario
3. Testing the get_llm_params function with user-provided keys
"""

import os
import sys
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

# Add the project root to the path
sys.path.insert(0, "/Users/vinh.nguyenxuan/Developer/learn-by-doing/VT.ai")


def test_byok_authentication_priority():
    """Test the authentication priority chain for BYOK functionality."""

    # Import after path setup
    from vtai.utils.llm_providers_config import get_llm_params

    print("🧪 Testing BYOK Authentication Priority Chain")
    print("=" * 50)

    # Test Case 1: No keys available (should fail gracefully)
    print("\n1️⃣ Testing with no API keys available...")

    # Clear environment variables temporarily
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    original_anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]

    try:
        # Mock Chainlit user session to simulate no user_env
        with patch("vtai.utils.llm_providers_config.cl") as mock_cl:
            mock_user_session = MagicMock()
            mock_user_session.get.return_value = None  # No user_env
            mock_cl.user_session = mock_user_session

            params = get_llm_params("openai/gpt-4o-mini")
            print(f"   Result: {params}")

            if not params.get("api_key"):
                print("   ✅ Correctly returned no API key when none available")
            else:
                print("   ❌ Unexpected API key found")

        # Test Case 2: User provides API key via user_env (BYOK)
        print("\n2️⃣ Testing with user_env API key (BYOK)...")

        mock_user_env = {
            "OPENAI_API_KEY": "sk-user-provided-key-12345",
            "ANTHROPIC_API_KEY": "sk-ant-user-provided-key-67890",
        }

        with patch("vtai.utils.llm_providers_config.cl") as mock_cl:
            mock_user_session = MagicMock()
            mock_user_session.get.return_value = mock_user_env  # User provided keys
            mock_cl.user_session = mock_user_session

            params = get_llm_params("openai/gpt-4o-mini")
            print(f"   Result: {params}")

            if params.get("api_key") == "sk-user-provided-key-12345":
                print("   ✅ Correctly used user_env API key (BYOK working!)")
            else:
                print(f"   ❌ Expected user key, got: {params.get('api_key')}")

        # Test Case 3: Test with user_keys parameter (secondary priority)
        print("\n3️⃣ Testing with user_keys parameter...")

        user_keys = {"openai": "sk-user-keys-parameter-999"}

        with patch("vtai.utils.llm_providers_config.cl") as mock_cl:
            mock_user_session = MagicMock()
            mock_user_session.get.return_value = None  # No user_env
            mock_cl.user_session = mock_user_session

            params = get_llm_params("openai/gpt-4o-mini", user_keys=user_keys)
            print(f"   Result: {params}")

            if params.get("api_key") == "sk-user-keys-parameter-999":
                print("   ✅ Correctly used user_keys parameter")
            else:
                print(f"   ❌ Expected user_keys value, got: {params.get('api_key')}")

        # Test Case 4: Test priority order (user_env should override user_keys)
        print("\n4️⃣ Testing priority order (user_env > user_keys)...")

        mock_user_env = {"OPENAI_API_KEY": "sk-user-env-priority"}
        user_keys = {"openai": "sk-user-keys-lower-priority"}

        with patch("vtai.utils.llm_providers_config.cl") as mock_cl:
            mock_user_session = MagicMock()
            mock_user_session.get.return_value = mock_user_env
            mock_cl.user_session = mock_user_session

            params = get_llm_params("openai/gpt-4o-mini", user_keys=user_keys)
            print(f"   Result: {params}")

            if params.get("api_key") == "sk-user-env-priority":
                print("   ✅ Correctly prioritized user_env over user_keys")
            else:
                print(f"   ❌ Expected user_env priority, got: {params.get('api_key')}")

        # Test Case 5: Test different providers
        print("\n5️⃣ Testing different providers...")

        mock_user_env = {
            "ANTHROPIC_API_KEY": "sk-ant-claude-key",
            "GEMINI_API_KEY": "gemini-key-123",
        }

        with patch("vtai.utils.llm_providers_config.cl") as mock_cl:
            mock_user_session = MagicMock()
            mock_user_session.get.return_value = mock_user_env
            mock_cl.user_session = mock_user_session

            # Test Anthropic
            params = get_llm_params("anthropic/claude-3-haiku-20240307")
            print(f"   Anthropic result: {params}")

            if params.get("api_key") == "sk-ant-claude-key":
                print("   ✅ Anthropic BYOK working")
            else:
                print(f"   ❌ Anthropic BYOK failed: {params.get('api_key')}")

            # Test Gemini
            params = get_llm_params("gemini/gemini-1.5-flash")
            print(f"   Gemini result: {params}")

            if params.get("api_key") == "gemini-key-123":
                print("   ✅ Gemini BYOK working")
            else:
                print(f"   ❌ Gemini BYOK failed: {params.get('api_key')}")

    finally:
        # Restore original environment variables
        if original_openai_key:
            os.environ["OPENAI_API_KEY"] = original_openai_key
        if original_anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = original_anthropic_key

    print("\n" + "=" * 50)
    print("🎉 BYOK Authentication Priority Test Complete!")
    print("\nThe authentication chain works as follows:")
    print("1. user_env (from Chainlit user_env config) - HIGHEST PRIORITY")
    print("2. user_keys (from function parameter) - MEDIUM PRIORITY")
    print("3. Environment variables - LOWEST PRIORITY")


def test_chainlit_config_detection():
    """Test if the Chainlit config has user_env properly configured."""

    print("\n🔧 Testing Chainlit Configuration")
    print("=" * 50)

    config_path = (
        "/Users/vinh.nguyenxuan/Developer/learn-by-doing/VT.ai/.chainlit/config.toml"
    )

    try:
        with open(config_path, "r") as f:
            config_content = f.read()

        if "[user_env]" in config_content:
            print("✅ user_env section found in config.toml")

            # Check for specific API key prompts
            api_keys_to_check = [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
                "OPENROUTER_API_KEY",
            ]

            for key in api_keys_to_check:
                if key in config_content:
                    print(f"   ✅ {key} configured for user prompt")
                else:
                    print(f"   ❌ {key} missing from user_env config")
        else:
            print("❌ user_env section not found in config.toml")

    except FileNotFoundError:
        print("❌ config.toml file not found")
    except Exception as e:
        print(f"❌ Error reading config.toml: {e}")


if __name__ == "__main__":
    print("🚀 VT.ai BYOK Functionality Test")
    print("=" * 50)

    # Test the authentication priority chain
    test_byok_authentication_priority()

    # Test the Chainlit configuration
    test_chainlit_config_detection()

    print("\n📋 Summary:")
    print("If all tests pass, BYOK functionality is properly implemented!")
    print("Users will be prompted for API keys when they first use models")
    print("that require providers they haven't configured yet.")
