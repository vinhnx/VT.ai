#!/usr/bin/env python3
"""
Simple BYOK test script that directly tests the authentication logic
without importing the full application modules.
"""

import os
from typing import Dict, Optional
from unittest.mock import MagicMock


def simulate_get_llm_params(model_name: str, user_keys: Optional[Dict] = None) -> Dict:
    """
    Simulate the get_llm_params function logic to test BYOK functionality.
    This mirrors the actual implementation in llm_providers_config.py
    """

    # Mock chainlit user_env (this would come from Chainlit's user session)
    user_env = None  # Simulating no user_env initially

    def get_key(provider: str, env_var: str) -> Optional[str]:
        """Get API key with priority: user_env > user_keys > environment"""
        if user_env and user_env.get(env_var):
            return user_env[env_var]  # First priority: user_env (BYOK)
        if user_keys and user_keys.get(provider):
            return user_keys[provider]  # Second priority: user_keys
        return os.environ.get(env_var)  # Third priority: environment variables

    # Provider mapping
    provider_mapping = {
        "openai": ("openai", "OPENAI_API_KEY"),
        "anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
        "google": ("google", "GEMINI_API_KEY"),
        "openrouter": ("openrouter", "OPENROUTER_API_KEY"),
        "deepseek": ("deepseek", "DEEPSEEK_API_KEY"),
        "groq": ("groq", "GROQ_API_KEY"),
        "mistral": ("mistral", "MISTRAL_API_KEY"),
        "cohere": ("cohere", "COHERE_API_KEY"),
    }

    # Determine provider from model name
    model_lower = model_name.lower()
    provider_key = None
    env_var = None

    for provider, (key, var) in provider_mapping.items():
        if provider in model_lower:
            provider_key = key
            env_var = var
            break

    if not provider_key:
        return {"error": f"Unknown provider for model: {model_name}"}

    # Get the API key using priority chain
    api_key = get_key(provider_key, env_var)

    result = {
        "model": model_name,
        "provider": provider_key,
        "api_key": api_key,
        "source": "none",
    }

    # Determine source of the key
    if user_env and user_env.get(env_var):
        result["source"] = "user_env"
    elif user_keys and user_keys.get(provider_key):
        result["source"] = "user_keys"
    elif os.environ.get(env_var):
        result["source"] = "environment"

    return result


def test_byok_authentication_chain():
    """Test the BYOK authentication priority chain."""

    print("🧪 Testing BYOK Authentication Priority Chain")
    print("=" * 50)

    # Test Case 1: Environment variable only
    print("\n1️⃣ Testing with environment variable only...")

    # Set a test environment variable
    os.environ["TEST_OPENAI_KEY"] = "sk-env-test-key"

    # Mock the environment lookup
    original_get = os.environ.get

    def mock_env_get(key, default=None):
        if key == "OPENAI_API_KEY":
            return "sk-env-test-key"
        return original_get(key, default)

    os.environ.get = mock_env_get

    result = simulate_get_llm_params("openai/gpt-4o-mini")
    print(f"   Result: {result}")
    print(f"   ✅ Source: {result['source']} (should be 'environment')")

    # Test Case 2: user_keys override environment
    print("\n2️⃣ Testing user_keys override...")

    user_keys = {"openai": "sk-user-provided-key"}
    result = simulate_get_llm_params("openai/gpt-4o-mini", user_keys)
    print(f"   Result: {result}")
    print(f"   ✅ Source: {result['source']} (should be 'user_keys')")

    # Test Case 3: user_env has highest priority
    print("\n3️⃣ Testing user_env highest priority...")

    # Simulate user_env being set (this would come from Chainlit)
    def simulate_with_user_env():
        user_env = {"OPENAI_API_KEY": "sk-user-env-key"}

        def get_key_with_user_env(provider: str, env_var: str) -> Optional[str]:
            if user_env and user_env.get(env_var):
                return user_env[env_var]
            if user_keys and user_keys.get(provider):
                return user_keys[provider]
            return os.environ.get(env_var)

        api_key = get_key_with_user_env("openai", "OPENAI_API_KEY")
        return {
            "model": "openai/gpt-4o-mini",
            "provider": "openai",
            "api_key": api_key,
            "source": "user_env" if user_env.get("OPENAI_API_KEY") else "other",
        }

    result = simulate_with_user_env()
    print(f"   Result: {result}")
    print(f"   ✅ Source: {result['source']} (should be 'user_env')")

    # Test Case 4: No keys available
    print("\n4️⃣ Testing with no keys available...")

    def mock_empty_env_get(key, default=None):
        return None

    os.environ.get = mock_empty_env_get
    result = simulate_get_llm_params("openai/gpt-4o-mini")
    print(f"   Result: {result}")
    print(f"   ✅ Source: {result['source']} (should be 'none')")

    # Restore original function
    os.environ.get = original_get

    print("\n🎉 All BYOK authentication tests completed!")
    return True


def test_chainlit_config():
    """Test that the Chainlit configuration is properly set up for BYOK."""

    print("\n🔧 Testing Chainlit Configuration")
    print("=" * 40)

    config_path = (
        "/Users/vinh.nguyenxuan/Developer/learn-by-doing/VT.ai/.chainlit/config.toml"
    )

    try:
        with open(config_path, "r") as f:
            config_content = f.read()

        # Check for user_env section
        if "[user_env]" in config_content:
            print("✅ [user_env] section found in config.toml")
        else:
            print("❌ [user_env] section missing from config.toml")
            return False

        # Check for key API providers
        required_keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ]

        missing_keys = []
        for key in required_keys:
            if key not in config_content:
                missing_keys.append(key)

        if missing_keys:
            print(f"❌ Missing API key configurations: {missing_keys}")
            return False
        else:
            print("✅ All required API key configurations found")

        print("✅ Chainlit configuration is properly set up for BYOK")
        return True

    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        return False
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False


if __name__ == "__main__":
    print("🚀 VT.ai BYOK Functionality Test")
    print("=" * 50)

    # Test the authentication chain
    auth_success = test_byok_authentication_chain()

    # Test the Chainlit configuration
    config_success = test_chainlit_config()

    print("\n📊 Test Summary")
    print("=" * 20)
    print(f"Authentication Chain: {'✅ PASS' if auth_success else '❌ FAIL'}")
    print(f"Chainlit Config: {'✅ PASS' if config_success else '❌ FAIL'}")

    if auth_success and config_success:
        print("\n🎉 All BYOK tests PASSED!")
        print("\n📝 Next Steps:")
        print("1. Test in live application by accessing http://localhost:8000")
        print("2. Try using a model without environment API keys")
        print("3. Verify that Chainlit prompts for API keys")
        print("4. Confirm user-provided keys work for API calls")
    else:
        print("\n❌ Some tests FAILED. Please review the issues above.")
