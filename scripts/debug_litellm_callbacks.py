#!/usr/bin/env python
"""
Debug test script for LiteLLM callbacks to Supabase.

This script focuses specifically on ensuring the callback function is properly
registered and executed.
"""

import inspect
import os
import sys
import time
from typing import Any, Dict, List, Optional

import dotenv
import litellm
from supabase import Client, create_client

# Add parent directory to path to import vtai modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for maximum detail
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vt.ai.debug")

# Load environment variables
dotenv.load_dotenv()


# Define a simple test callback to verify the callback mechanism
class TestCallback:
    def __init__(self):
        self.called = False
        logger.info("TestCallback initialized")

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        self.called = True
        logger.info("TestCallback.log_success_event called!")
        logger.info(
            f"Response contains token usage: {getattr(response_obj, 'usage', None)}"
        )

    def log_failure_event(
        self, kwargs: Dict[str, Any], error: Any, start_time: float, end_time: float
    ) -> None:
        self.called = True
        logger.info("TestCallback.log_failure_event called!")
        logger.info(f"Error: {error}")


def get_supabase_client(use_service_key: bool = True) -> Optional[Client]:
    """
    Initialize the Supabase client using environment variables.

    Args:
        use_service_key: Whether to use the service key instead of the regular key

    Returns:
        Initialized Supabase client or None if initialization fails
    """
    supabase_url = os.environ.get("SUPABASE_URL")

    # Use service key if requested, otherwise use regular key
    if use_service_key:
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not supabase_key:
            logger.warning(
                "SUPABASE_SERVICE_KEY not found, falling back to SUPABASE_KEY"
            )
            supabase_key = os.environ.get("SUPABASE_KEY")
    else:
        supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        logger.error(
            "Required Supabase environment variables not set. Please set SUPABASE_URL and either SUPABASE_KEY or SUPABASE_SERVICE_KEY."
        )
        return None

    try:
        client = create_client(supabase_url, supabase_key)
        if use_service_key and os.environ.get("SUPABASE_SERVICE_KEY") == supabase_key:
            logger.info(
                "Supabase client initialized with service key (elevated privileges)."
            )
        else:
            logger.info("Supabase client initialized with regular key.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return None


def inspect_litellm_callbacks():
    """
    Inspect LiteLLM's callback mechanism to understand how it's configured.
    """
    logger.info("Inspecting LiteLLM callback mechanism...")

    # Check if handlers are registered
    if hasattr(litellm, "callbacks"):
        logger.info(f"litellm.callbacks = {litellm.callbacks}")
        for i, callback in enumerate(litellm.callbacks):
            logger.info(f"Callback {i}: {callback.__class__.__name__}")
    else:
        logger.warning("litellm.callbacks does not exist")

    # Check success callback attribute
    if hasattr(litellm, "success_callback"):
        logger.info(f"litellm.success_callback = {litellm.success_callback}")
    else:
        logger.warning("litellm.success_callback does not exist")

    # Check failure callback attribute
    if hasattr(litellm, "failure_callback"):
        logger.info(f"litellm.failure_callback = {litellm.failure_callback}")
    else:
        logger.warning("litellm.failure_callback does not exist")

    # Examine the actual function that handles callbacks
    if hasattr(litellm.utils, "handle_success"):
        logger.info("Examining litellm.utils.handle_success function...")
        handle_success_source = inspect.getsource(litellm.utils.handle_success)
        logger.debug(
            f"handle_success source:\n{handle_success_source[:500]}..."
        )  # Show first 500 chars
    else:
        logger.warning("litellm.utils.handle_success does not exist")


def test_callback_functionality():
    """
    Test LiteLLM callback functionality with a simple test callback.
    """
    logger.info("Testing callback functionality...")

    # First, clear any existing callbacks
    litellm.callbacks = []
    if hasattr(litellm, "success_callback"):
        litellm.success_callback = []
    if hasattr(litellm, "failure_callback"):
        litellm.failure_callback = []

    # Create and register our test callback
    test_callback = TestCallback()
    litellm.callbacks = [test_callback]

    if hasattr(litellm, "success_callback"):
        litellm.success_callback = [test_callback]
    if hasattr(litellm, "failure_callback"):
        litellm.failure_callback = [test_callback]

    # Verify it was registered
    logger.info(f"Registered test callback, litellm.callbacks = {litellm.callbacks}")

    # Test model - using a small free model to avoid costs
    model = "openrouter/qwen/qwen3-0.6b-04-28:free"

    # Check if OPENROUTER_API_KEY is set
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error(
            "OPENROUTER_API_KEY not set in environment variables. Required for the test model."
        )
        return False

    # Make a test request
    try:
        logger.info(f"Making test completion request with model: {model}")
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hello, this is a test message."}],
            max_tokens=10,
            user="callback-test-user",
        )

        logger.info(f"Response received: {response.choices[0].message.content}")
        logger.info(f"Token usage: {response.usage}")

        # Wait for any async callbacks to complete
        logger.info("Waiting for callbacks to complete...")
        time.sleep(2)

        # Check if our callback was called
        if test_callback.called:
            logger.info("SUCCESS: Test callback was called!")
            return True
        else:
            logger.error("FAILURE: Test callback was not called!")

            # Try to diagnose the issue
            inspect_litellm_callbacks()
            return False

    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    logger.info("Starting callback debug test...")

    # First, check LiteLLM's callback mechanism
    inspect_litellm_callbacks()

    # Then, run our test
    if test_callback_functionality():
        logger.info("Test completed successfully.")
        sys.exit(0)
    else:
        logger.error("Test failed.")
        sys.exit(1)
