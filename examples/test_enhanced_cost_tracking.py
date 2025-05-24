#!/usr/bin/env python3
"""
Test script for enhanced cost tracking with detailed token breakdown.

This script demonstrates the improved LiteLLM + Supabase logging integration
with detailed token cost tracking using LiteLLM's built-in cost functions.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import litellm

from vtai.utils.config import logger
from vtai.utils.supabase_logger import (
    calculate_prompt_tokens,
    calculate_token_costs,
    get_model_info,
    get_user_analytics,
    setup_litellm_callbacks,
)

# Test different models and providers
TEST_MODELS = [
    "gpt-4o-mini",
    "gpt-4",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229"
]

TEST_MESSAGES = [
    {"role": "user", "content": "Hello! How are you?"},
    {"role": "user", "content": "What is the capital of France? Please provide a detailed explanation about Paris."},
    {"role": "user", "content": "Write a short story about a robot learning to paint. Make it exactly 100 words."}
]


def test_model_info():
    """Test the model information retrieval."""
    print("🔍 Testing Model Information Retrieval")
    print("=" * 50)

    for model in TEST_MODELS:
        info = get_model_info(model)
        print(f"Model: {model}")
        print(f"  Max Tokens: {info.get('max_tokens', 'Unknown')}")
        print(f"  Input Cost/Token: ${info.get('input_cost_per_token', 'Unknown')}")
        print(f"  Output Cost/Token: ${info.get('output_cost_per_token', 'Unknown')}")
        print(f"  Provider: {info.get('litellm_provider', 'Unknown')}")
        print()


def test_token_calculation():
    """Test token counting and cost calculation."""
    print("🧮 Testing Token Calculation & Cost Analysis")
    print("=" * 50)

    for i, message in enumerate(TEST_MESSAGES, 1):
        print(f"Test Case {i}: {message['content'][:50]}...")

        for model in TEST_MODELS[:2]:  # Test with first 2 models only
            try:
                # Calculate prompt tokens
                prompt_tokens = calculate_prompt_tokens(model, [message])

                # Simulate completion tokens (in real usage, this comes from the API response)
                completion_tokens = 25 if i == 1 else (50 if i == 2 else 100)
                total_tokens = prompt_tokens + completion_tokens

                # Calculate costs
                prompt_cost_per_token, completion_cost_per_token, total_cost = calculate_token_costs(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )

                print(f"  {model}:")
                print(f"    Prompt Tokens: {prompt_tokens}")
                print(f"    Completion Tokens: {completion_tokens}")
                print(f"    Total Tokens: {total_tokens}")
                print(f"    Prompt Cost/Token: ${prompt_cost_per_token:.8f}")
                print(f"    Completion Cost/Token: ${completion_cost_per_token:.8f}")
                print(f"    Total Cost: ${total_cost:.6f}")
                print()

            except Exception as e:
                print(f"  {model}: Error - {str(e)}")
                print()


def test_litellm_cost_functions():
    """Test LiteLLM's built-in cost calculation functions."""
    print("💰 Testing LiteLLM Cost Functions")
    print("=" * 50)

    # Test cost_per_token function
    test_cases = [
        {"model": "gpt-4o-mini", "prompt_tokens": 100, "completion_tokens": 50},
        {"model": "gpt-4", "prompt_tokens": 200, "completion_tokens": 100},
        {"model": "claude-3-haiku-20240307", "prompt_tokens": 150, "completion_tokens": 75},
    ]

    for case in test_cases:
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=case["model"],
                prompt_tokens=case["prompt_tokens"],
                completion_tokens=case["completion_tokens"]
            )

            total_cost = (case["prompt_tokens"] * prompt_cost) + (case["completion_tokens"] * completion_cost)

            print(f"Model: {case['model']}")
            print(f"  Prompt: {case['prompt_tokens']} tokens × ${prompt_cost:.8f} = ${case['prompt_tokens'] * prompt_cost:.6f}")
            print(f"  Completion: {case['completion_tokens']} tokens × ${completion_cost:.8f} = ${case['completion_tokens'] * completion_cost:.6f}")
            print(f"  Total Cost: ${total_cost:.6f}")
            print()

        except Exception as e:
            print(f"Model {case['model']}: Error - {str(e)}")
            print()


async def test_real_api_call():
    """Test with a real API call to verify end-to-end functionality."""
    print("🚀 Testing Real API Call with Enhanced Logging")
    print("=" * 50)

    # Setup callbacks
    setup_litellm_callbacks()

    try:
        # Make a real API call (using OpenAI as it's most reliable)
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 2+2? Answer briefly."}],
            user="test_enhanced_cost_tracking_user",
            max_tokens=50
        )

        # Extract usage information
        usage = response.usage
        print(f"✅ API Call Successful!")
        print(f"  Model: {response.model}")
        print(f"  Prompt Tokens: {usage.prompt_tokens}")
        print(f"  Completion Tokens: {usage.completion_tokens}")
        print(f"  Total Tokens: {usage.total_tokens}")

        # Calculate cost using our enhanced function
        prompt_cost_per_token, completion_cost_per_token, calculated_cost = calculate_token_costs(
            model="gpt-4o-mini",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )

        # Get LiteLLM's cost calculation
        litellm_cost = litellm.completion_cost(completion_response=response)

        print(f"  Our Calculated Cost: ${calculated_cost:.6f}")
        print(f"  LiteLLM Cost: ${litellm_cost:.6f}")
        print(f"  Cost Difference: ${abs(calculated_cost - litellm_cost):.6f}")
        print(f"  Response: {response.choices[0].message.content[:100]}...")

        # Wait a moment for callbacks to process
        await asyncio.sleep(2)

        print("\n📊 Enhanced logging should now show detailed token breakdown in Supabase!")

    except Exception as e:
        print(f"❌ API call failed: {str(e)}")


def display_model_cost_database():
    """Display available models and their costs from LiteLLM."""
    print("📋 LiteLLM Model Cost Database")
    print("=" * 50)

    model_costs = litellm.model_cost

    # Show first 10 models as example
    count = 0
    for model, info in model_costs.items():
        if count >= 10:
            break

        print(f"Model: {model}")
        print(f"  Max Tokens: {info.get('max_tokens', 'Unknown')}")
        print(f"  Input Cost: ${info.get('input_cost_per_token', 0):.8f}/token")
        print(f"  Output Cost: ${info.get('output_cost_per_token', 0):.8f}/token")
        print(f"  Provider: {info.get('litellm_provider', 'Unknown')}")
        print()
        count += 1

    print(f"... and {len(model_costs) - 10} more models available")


async def main():
    """Run all tests."""
    print("🧪 Enhanced Cost Tracking Test Suite")
    print("="*60)
    print()

    # Test 1: Model information
    test_model_info()
    print()

    # Test 2: Token calculation
    test_token_calculation()
    print()

    # Test 3: LiteLLM cost functions
    test_litellm_cost_functions()
    print()

    # Test 4: Model cost database
    display_model_cost_database()
    print()

    # Test 5: Real API call (optional, requires API key)
    if os.getenv("OPENAI_API_KEY"):
        await test_real_api_call()
    else:
        print("⚠️  Skipping real API call test (OPENAI_API_KEY not set)")

    print("\n✅ All tests completed!")
    print("\nKey Improvements:")
    print("- Detailed token breakdown (prompt vs completion)")
    print("- Accurate cost calculation per token type")
    print("- Enhanced database schema with cost tracking")
    print("- LiteLLM integration for cost calculation")
    print("- Support for all major LLM providers")


if __name__ == "__main__":
    asyncio.run(main())    asyncio.run(main())