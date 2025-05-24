#!/usr/bin/env python3
"""
Demo script showing enhanced cost tracking without requiring API keys.

This demonstrates the improved cost calculation and database logging capabilities.
"""

import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtai.utils.config import logger
from vtai.utils.supabase_logger import (
    calculate_token_costs,
    get_model_info,
    get_user_analytics,
    log_request_to_supabase,
)


def demo_enhanced_cost_tracking():
    """Demonstrate enhanced cost tracking capabilities."""
    print("🚀 Enhanced LiteLLM + Supabase Cost Tracking Demo")
    print("=" * 55)

    # Test data simulating a real API response
    test_scenarios = [
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello! How are you today?"}],
            "prompt_tokens": 13,
            "completion_tokens": 25,
            "response": {"choices": [{"message": {"content": "I'm doing well, thank you for asking!"}}]},
            "user": "demo_user_enhanced_tracking"
        },
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
            "prompt_tokens": 22,
            "completion_tokens": 85,
            "response": {"choices": [{"message": {"content": "Quantum computing uses quantum mechanical phenomena..."}}]},
            "user": "demo_user_enhanced_tracking"
        },
        {
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "Write a haiku about coding."}],
            "prompt_tokens": 15,
            "completion_tokens": 30,
            "response": {"choices": [{"message": {"content": "Code flows like water\\nBugs surface, then disappear\\nLogic finds its way"}}]},
            "user": "demo_user_enhanced_tracking"
        }
    ]

    print("📊 Processing Test Scenarios with Enhanced Cost Tracking...")
    print()

    total_cost = 0.0
    total_tokens = 0

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"Scenario {i}: {scenario['model']}")
        print(f"  Prompt: {scenario['messages'][0]['content']}")

        # Calculate detailed costs
        prompt_cost_per_token, completion_cost_per_token, calculated_cost = calculate_token_costs(
            model=scenario["model"],
            prompt_tokens=scenario["prompt_tokens"],
            completion_tokens=scenario["completion_tokens"],
            total_tokens=scenario["prompt_tokens"] + scenario["completion_tokens"]
        )

        # Display detailed breakdown
        print(f"  📝 Token Breakdown:")
        print(f"    Prompt Tokens: {scenario['prompt_tokens']} × ${prompt_cost_per_token:.8f} = ${scenario['prompt_tokens'] * prompt_cost_per_token:.6f}")
        print(f"    Completion Tokens: {scenario['completion_tokens']} × ${completion_cost_per_token:.8f} = ${scenario['completion_tokens'] * completion_cost_per_token:.6f}")
        print(f"    Total Cost: ${calculated_cost:.6f}")
        print(f"    Total Tokens: {scenario['prompt_tokens'] + scenario['completion_tokens']}")

        # Log to Supabase with enhanced data
        try:
            log_request_to_supabase(
                model=scenario["model"],
                messages=scenario["messages"],
                response=scenario["response"],
                end_user=scenario["user"],
                status="success",
                response_time=0.5 + (i * 0.1),  # Simulated response time
                total_cost=calculated_cost,
                litellm_call_id=f"demo_call_{i}_{datetime.now().strftime('%H%M%S')}",
                user_profile_id=scenario["user"],
                tokens_used=scenario["prompt_tokens"] + scenario["completion_tokens"],
                provider=scenario["model"].split("/")[0] if "/" in scenario["model"] else "openai",
                prompt_tokens=scenario["prompt_tokens"],
                completion_tokens=scenario["completion_tokens"],
                prompt_cost_per_token=prompt_cost_per_token,
                completion_cost_per_token=completion_cost_per_token
            )
            print(f"  ✅ Logged to Supabase with enhanced cost tracking")
        except Exception as e:
            print(f"  ❌ Error logging to Supabase: {str(e)}")

        total_cost += calculated_cost
        total_tokens += scenario["prompt_tokens"] + scenario["completion_tokens"]
        print()

    print("📈 Summary:")
    print(f"  Total Scenarios: {len(test_scenarios)}")
    print(f"  Total Tokens: {total_tokens}")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Average Cost per Token: ${total_cost/total_tokens:.8f}")

    # Try to get user analytics
    print("\n📊 User Analytics (if available):")
    try:
        analytics = get_user_analytics("demo_user_enhanced_tracking")
        if analytics:
            print(f"  User: {analytics.get('user_id', 'Unknown')}")
            print(f"  Total Requests: {analytics.get('total_requests', 0)}")
            print(f"  Total Tokens: {analytics.get('total_tokens', 0)}")
            print(f"  Total Cost: ${analytics.get('total_cost', 0):.6f}")
            print(f"  Average Tokens per Request: {analytics.get('avg_tokens_per_request', 0):.1f}")
        else:
            print("  No analytics available yet (may take a moment to process)")
    except Exception as e:
        print(f"  Error retrieving analytics: {str(e)}")


def show_model_pricing():
    """Show pricing for popular models."""
    print("\n💰 Popular Model Pricing Information")
    print("=" * 40)

    popular_models = [
        "gpt-4o-mini",
        "gpt-4",
        "gpt-4-turbo",
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229"
    ]

    for model in popular_models:
        info = get_model_info(model)
        input_cost = info.get('input_cost_per_token', 0)
        output_cost = info.get('output_cost_per_token', 0)

        if input_cost > 0:
            print(f"{model}:")
            print(f"  Input:  ${input_cost:.8f}/token (${input_cost * 1000:.5f}/1K tokens)")
            print(f"  Output: ${output_cost:.8f}/token (${output_cost * 1000:.5f}/1K tokens)")
            print(f"  Max Tokens: {info.get('max_tokens', 'Unknown')}")
        else:
            print(f"{model}: Pricing not available")
        print()


def main():
    """Run the demo."""
    demo_enhanced_cost_tracking()
    show_model_pricing()

    print("\n🎉 Enhanced Cost Tracking Features:")
    print("✅ Detailed token breakdown (prompt vs completion)")
    print("✅ Accurate per-token cost calculation")
    print("✅ Enhanced Supabase logging with cost data")
    print("✅ Support for all major LLM providers")
    print("✅ Real-time cost tracking and analytics")
    print("✅ Database triggers for automatic token counting")


if __name__ == "__main__":
    main()    main()