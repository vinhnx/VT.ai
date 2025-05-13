#!/usr/bin/env python
"""
Test script to verify the 'APIResponse[~_ReturnT] can't be used in 'await' expression' error is fixed.
This script performs direct tests of the token tracking functionality without requiring a full application startup.
"""

import json
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

# Import the token usage tracking components
from vtai.utils.litellm_callbacks import (
    VTAISupabaseHandler,
    initialize_litellm_callbacks,
)
from vtai.utils.usage_logger import log_usage_to_supabase

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vt.ai.test")

# Load environment variables
dotenv.load_dotenv()


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


async def test_manual_logging():
    """
    Test the direct usage logging function without using LiteLLM callbacks.
    """
    logger.info("Testing direct manual usage logging...")

    # Initialize Supabase client
    supabase_client = get_supabase_client(use_service_key=True)
    if not supabase_client:
        logger.error("Failed to initialize Supabase client. Exiting.")
        return False

    # Test user ID and session
    test_user_id = "test-user-" + str(int(time.time()))
    test_session_id = "test-session-" + str(int(time.time()))

    try:
        # Test manual logging
        await log_usage_to_supabase(
            user_id=test_user_id,
            session_id=test_session_id,
            model_name="test-model",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost=0.001,
        )

        logger.info("Manual logging completed successfully!")

        # Verify it worked by querying the database
        logger.info("Verifying log entry...")
        try:
            # Query without awaiting
            query = (
                supabase_client.table("request_logs")
                .select("*")
                .eq("end_user", test_user_id)
            )
            result = query.execute()

            if result.data and len(result.data) > 0:
                logger.info(f"Found {len(result.data)} manual log entries!")
                logger.info(f"First entry: {json.dumps(result.data[0], indent=2)}")
                return True
            else:
                logger.warning("No manual log entries found!")
                return False

        except Exception as e:
            logger.error(f"Error verifying log entry: {e}")
            return False

    except Exception as e:
        logger.error(f"Error in manual usage logging test: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def test_litellm_callbacks():
    """
    Test LiteLLM callbacks with a simple completion request.
    """
    logger.info("Testing LiteLLM callbacks...")

    # Initialize Supabase client
    supabase_client = get_supabase_client(use_service_key=True)
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

        return True

    except Exception as e:
        logger.error(f"Error in LiteLLM callback test: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Test fixes for token usage tracking.")
    parser.add_argument(
        "--manual-only", action="store_true", help="Only test manual logging"
    )
    parser.add_argument(
        "--litellm-only", action="store_true", help="Only test LiteLLM callbacks"
    )
    args = parser.parse_args()

    success = True

    # Test manual logging if requested or if testing everything
    if args.manual_only or not (args.manual_only or args.litellm_only):
        manual_result = asyncio.run(test_manual_logging())
        if manual_result:
            logger.info("✅ Manual logging test passed!")
        else:
            logger.error("❌ Manual logging test failed!")
            success = False

    # Test LiteLLM callbacks if requested or if testing everything
    if args.litellm_only or not (args.manual_only or args.litellm_only):
        litellm_result = test_litellm_callbacks()
        if litellm_result:
            logger.info("✅ LiteLLM callbacks test passed!")
        else:
            logger.error("❌ LiteLLM callbacks test failed!")
            success = False

    if success:
        logger.info("All tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("Some tests failed!")
        sys.exit(1)
