"""
Example implementation of LiteLLM with Supabase callbacks for request logging.

This example demonstrates how to:
1. Set up environment variables for Supabase and OpenAI
2. Configure LiteLLM success and failure callbacks
3. Make completion calls that get logged to Supabase
"""

import os
from typing import Any, Dict, Optional

import litellm
from litellm import completion

# Set environment variables
# SUPABASE - Use the same environment variable names as the main app
os.environ["SUPABASE_URL"] = os.environ.get(
    "SUPABASE_URL", "https://moykyctcjahifdkmhybt.supabase.co"
)
# The supabase_logger checks for SUPABASE_KEY first, then SUPABASE_ANON_KEY
os.environ["SUPABASE_KEY"] = os.environ.get("SUPABASE_ANON_KEY", "your-supabase-key")

# LLM API KEY
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

# Import the existing Supabase logger from VT.ai
from vtai.utils.supabase_logger import log_request_to_supabase


def supabase_success_callback(
    kwargs: Dict[str, Any],
    completion_response: Any,
    start_time: float,
    end_time: float,
) -> None:
    """
    Success callback for LiteLLM to log successful requests to Supabase.

    Args:
            kwargs: Original arguments passed to completion
            completion_response: The response from LiteLLM
            start_time: Start time of the request
            end_time: End time of the request
    """
    try:
        # Extract relevant information
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        user = kwargs.get("user", "anonymous")

        # Calculate response time
        response_time = end_time - start_time

        # Extract response content
        response_content = None
        if hasattr(completion_response, "choices") and completion_response.choices:
            choice = completion_response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                response_content = {"content": choice.message.content}

        # Calculate cost if available
        total_cost = None
        if (
            hasattr(completion_response, "_hidden_params")
            and "response_cost" in completion_response._hidden_params
        ):
            total_cost = completion_response._hidden_params["response_cost"]

        # Get LiteLLM call ID
        litellm_call_id = getattr(completion_response, "id", None)

        # Log to Supabase
        log_request_to_supabase(
            model=model,
            messages=messages,
            response=response_content,
            end_user=user,
            status="success",
            response_time=response_time,
            total_cost=total_cost,
            litellm_call_id=litellm_call_id,
        )

        print(f"✅ Successfully logged request for model {model}")

    except Exception as e:
        print(f"❌ Error in success callback: {type(e).__name__}: {str(e)}")


def supabase_failure_callback(
    kwargs: Dict[str, Any],
    completion_response: Any,
    start_time: float,
    end_time: float,
) -> None:
    """
    Failure callback for LiteLLM to log failed requests to Supabase.

    Args:
            kwargs: Original arguments passed to completion
            completion_response: The error/exception that occurred
            start_time: Start time of the request
            end_time: End time of the request
    """
    try:
        # Extract relevant information
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        user = kwargs.get("user", "anonymous")

        # Calculate response time
        response_time = end_time - start_time

        # Extract error information
        error_info = {
            "error": str(completion_response),
            "error_type": type(completion_response).__name__,
        }

        # Log to Supabase
        log_request_to_supabase(
            model=model,
            messages=messages,
            response=None,
            end_user=user,
            status="failure",
            error=error_info,
            response_time=response_time,
        )

        print(f"❌ Logged failed request for model {model}: {error_info}")

    except Exception as e:
        print(f"❌ Error in failure callback: {type(e).__name__}: {str(e)}")


def setup_litellm_callbacks():
    """Set up LiteLLM callbacks for Supabase logging."""
    # Set callbacks
    litellm.success_callback = [supabase_success_callback]
    litellm.failure_callback = [supabase_failure_callback]

    print("🔧 LiteLLM callbacks configured for Supabase logging")


def main():
    """Main function to demonstrate LiteLLM with Supabase callbacks."""
    print("🚀 Starting LiteLLM Supabase callbacks example")

    # Setup callbacks
    setup_litellm_callbacks()

    # Good call - should succeed and get logged
    print("\n📞 Making successful OpenAI call...")
    try:
        response = completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
            user="vinhnx",  # identify users
        )
        print(f"✅ Success: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    # Bad call - expect this call to fail and get logged
    print("\n📞 Making bad call to test error logging...")
    try:
        response = completion(
            model="chatgpt-test",  # Invalid model
            messages=[
                {
                    "role": "user",
                    "content": "Hi 👋 - i'm a bad call to test error logging",
                }
            ],
        )
        print(f"✅ Unexpected success: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Expected error (will be logged): {type(e).__name__}: {str(e)}")

    # Another good call
    print("\n📞 Making another successful call...")
    try:
        response = completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
            user="ishaan22",  # identify users
        )
        print(f"✅ Success: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    print(
        "\n🎉 Example completed! Check your Supabase request_logs table for logged entries."
    )


if __name__ == "__main__":
    main()
