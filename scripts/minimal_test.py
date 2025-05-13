#!/usr/bin/env python
"""
Minimal test script to verify the fix for 'APIResponse[~_ReturnT] can't be used in 'await' expression'
"""

import logging
import os
import sys
import time

import dotenv
from supabase import create_client

# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vt.ai.minimal_test")

# Load environment variables
dotenv.load_dotenv()


def main():
    # Get Supabase credentials
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get(
        "SUPABASE_SERVICE_KEY", os.environ.get("SUPABASE_KEY")
    )

    if not supabase_url or not supabase_key:
        logger.error(
            "Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_SERVICE_KEY or SUPABASE_KEY."
        )
        return False

    # Initialize Supabase client
    try:
        client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return False

    # Create a test entry
    test_id = f"test-{int(time.time())}"
    test_data = {
        "model": "test-model",
        "messages": "[]",
        "response": "{}",
        "end_user": test_id,
        "status": "success",
        "response_time": 0.0,
        "total_cost": 0.0,
        "additional_details": '{"test": true}',
        "litellm_call_id": test_id,
    }

    try:
        # Create the query - NO AWAIT HERE
        logger.info("Creating Supabase query...")
        query = client.table("request_logs").insert(test_data)

        # Execute the query - NO AWAIT HERE
        logger.info("Executing query...")
        result = query.execute()

        # Check the result
        logger.info("Query executed. Checking result...")
        if hasattr(result, "data") and result.data:
            logger.info(f"Success! Inserted data with ID: {result.data[0].get('id')}")
            return True
        else:
            logger.warning(f"Query executed but no data returned: {result}")
            return False

    except Exception as e:
        logger.error(f"Error executing Supabase query: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ Test passed!")
        sys.exit(0)
    else:
        logger.error("❌ Test failed!")
        sys.exit(1)
        sys.exit(1)
