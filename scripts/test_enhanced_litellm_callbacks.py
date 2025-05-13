#!/usr/bin/env python
"""
Enhanced test script for LiteLLM callbacks to Supabase.

This script provides a more robust test for token usage tracking with LiteLLM,
including tests for edge cases and error handling.
"""

import os
import sys
import time
from typing import Optional

import dotenv
import litellm
from supabase import Client, create_client

# Add parent directory to path to import vtai modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure basic logging
import logging

# Import our custom callback handler
from vtai.utils.litellm_callbacks import (
    VTAISupabaseHandler,
    initialize_litellm_callbacks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vt.ai.test")

# Load environment variables
dotenv.load_dotenv()


def get_supabase_client(use_service_key: bool = False) -> Optional[Client]:
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


def test_litellm_callbacks(use_service_key: bool = False):
    """
    Test LiteLLM callbacks with a simple completion request.

    Args:
        use_service_key: Whether to use the service key for Supabase
    """
    supabase_client = get_supabase_client(use_service_key=use_service_key)
    if not supabase_client:
        logger.error("Failed to initialize Supabase client. Exiting.")
        return False

    # Clear any existing callbacks to avoid duplicates
    litellm.callbacks = []

    # Initialize our custom callbacks
    initialize_litellm_callbacks(supabase_client)

    if not litellm.callbacks:
        logger.error("No LiteLLM callbacks were registered. Exiting.")
        return False

    logger.info(f"Initialized {len(litellm.callbacks)} LiteLLM callbacks")

    # Test model - using a small free model to avoid costs
    model = "openrouter/qwen/qwen3-0.6b-04-28:free"

    # Check if OPENROUTER_API_KEY is set
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error(
            "OPENROUTER_API_KEY not set in environment variables. Required for the test model."
        )
        return False

    # Test user ID
    test_user_id = "test-user-" + str(int(time.time()))

    try:
        # Make a test completion request
        logger.info(f"Making test completion request with model: {model}")
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hello, this is a test message."}],
            max_tokens=10,
            user=test_user_id,
            session_id="test-session-" + str(int(time.time())),
        )

        # Print the response
        logger.info(f"Response: {response.choices[0].message.content}")

        # Print token usage
        token_usage = response.usage
        logger.info(f"Token usage: {token_usage}")

        # Wait a moment for the callbacks to complete
        logger.info("Waiting for callbacks to complete...")
        time.sleep(2)

        # Check if the usage was logged to Supabase
        logger.info("Checking if usage was logged to request_logs table...")
        try:
            request_logs = (
                supabase_client.table("request_logs")
                .select("*")
                .eq("end_user", test_user_id)
                .execute()
            )

            if request_logs.data:
                logger.info(
                    f"Found {len(request_logs.data)} records in request_logs table."
                )
                logger.info(f"First record: {request_logs.data[0]}")
            else:
                logger.warning("No records found in request_logs table.")
        except Exception as e:
            logger.error(f"Error checking request_logs table: {e}")

        # Test with an invalid model (should trigger the failure callback)
        logger.info("Testing with an invalid model to test failure callback...")
        try:
            litellm.completion(
                model="invalid-model",
                messages=[{"role": "user", "content": "This should fail."}],
                max_tokens=10,
                user=test_user_id,
                session_id="test-session-failure-" + str(int(time.time())),
            )
        except Exception as e:
            logger.info(f"Expected error with invalid model: {e}")

        # Wait a moment for the callbacks to complete
        logger.info("Waiting for failure callbacks to complete...")
        time.sleep(2)

        # Check if failure was logged
        logger.info("Checking if failure was logged to request_logs table...")
        try:
            failure_logs = (
                supabase_client.table("request_logs")
                .select("*")
                .eq("end_user", test_user_id)
                .eq("status", "failed")
                .execute()
            )

            if failure_logs.data:
                logger.info(
                    f"Found {len(failure_logs.data)} failure records in request_logs table."
                )
                logger.info(f"First failure record: {failure_logs.data[0]}")
            else:
                logger.warning("No failure records found in request_logs table.")
        except Exception as e:
            logger.error(f"Error checking failure logs in request_logs table: {e}")

        return True
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Test LiteLLM callbacks with Supabase."
    )
    parser.add_argument(
        "--service-key",
        action="store_true",
        help="Use Supabase service key (elevated privileges) for testing",
    )
    args = parser.parse_args()

    # Run the test with the specified key
    if test_litellm_callbacks(use_service_key=args.service_key):
        logger.info("Test completed successfully.")
        sys.exit(0)
    else:
        logger.error("Test failed.")
        sys.exit(1)
