#!/usr/bin/env python
"""
Simple test script to verify the token tracking fix.
This focuses solely on testing the Supabase query pattern for the request_logs table.
"""

import asyncio
import json
import logging
import os
import sys
import time

import dotenv
from supabase import create_client

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_request_logs_insert():
    """
    Test inserting data into the request_logs table with the fixed pattern.
    """
    # Load environment variables
    dotenv.load_dotenv()

    # Get Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        logger.error("Missing Supabase credentials in environment variables")
        return False

    # Create Supabase client
    try:
        client = create_client(supabase_url, supabase_key)
        logger.info("Connected to Supabase")
    except Exception as e:
        logger.error(f"Error connecting to Supabase: {e}")
        return False

    # Create test log entry
    test_id = f"test-{int(time.time())}"
    log_entry = {
        "model": "test-model",
        "messages": json.dumps([{"role": "user", "content": "Hello, world!"}]),
        "end_user": "00000000-0000-0000-0000-000000000000",  # Anonymous UUID
        "status": "success",
        "response_time": 0.5,
        "request_id": test_id,
        "additional_details": json.dumps(
            {"tokens_used": 123, "test_run": True, "timestamp": time.time()}
        ),
    }

    try:
        # Use the correct pattern: build query then execute
        logger.info(f"Inserting test log entry with ID: {test_id}")
        query = client.table("request_logs").insert(log_entry)
        result = query.execute()

        if result.data:
            logger.info(f"Successfully inserted log entry: {result.data}")
            return True
        else:
            logger.error(f"Failed to insert log entry: {result}")
            return False
    except Exception as e:
        logger.error(f"Error testing request_logs insert: {e}")
        return False


if __name__ == "__main__":
    # Run the test
    logger.info("Starting token tracking test...")
    result = asyncio.run(test_request_logs_insert())

    if result:
        logger.info("✅ Test PASSED: Successfully inserted into request_logs table")
        sys.exit(0)
    else:
        logger.error("❌ Test FAILED: Could not insert into request_logs table")
        sys.exit(1)
        sys.exit(1)
