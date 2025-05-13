#!/usr/bin/env python
"""
Validate that the necessary RLS policies are in place for token usage tracking.

This script checks the RLS policies for the request_logs and usage_logs tables
to ensure that LiteLLM token tracking callbacks can write to both tables.
"""

import logging
import os
import sys
from typing import Dict, List, Optional

import dotenv
from supabase import Client, create_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vt.ai.validator")

# Load environment variables
dotenv.load_dotenv()


def get_supabase_client() -> Optional[Client]:
    """
    Initialize the Supabase client using environment variables.

    Returns:
        Initialized Supabase client or None if initialization fails
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    # For RLS policy validation, we can use the service key
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get(
        "SUPABASE_KEY"
    )

    if not supabase_url or not supabase_key:
        logger.error(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_KEY) must be set in environment variables."
        )
        return None

    try:
        # Initialize client
        client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully.")
        return client
    except Exception as e:
        logger.error("Failed to initialize Supabase client: %s", e)
        return None


def check_table_exists(client: Client, table_name: str) -> bool:
    """
    Check if a table exists in the database.

    Args:
        client: Initialized Supabase client
        table_name: Name of the table to check

    Returns:
        True if the table exists, False otherwise
    """
    try:
        # Try to select a single row from the table instead of using information_schema
        response = client.table(table_name).select("*").limit(1).execute()

        logger.info("✅ Table '%s' exists", table_name)
        return True
    except Exception as e:
        if "relation" in str(e) and "does not exist" in str(e):
            logger.error("❌ Table '%s' does not exist", table_name)
            print(f"\nTo create the {table_name} table, run:")
            print(f"python scripts/create_request_logs_table.py")
            return False
        else:
            logger.error("Error checking if table %s exists: %s", table_name, e)
            return False


def check_policies(client: Client, table_name: str) -> bool:
    """
    Check the RLS policies for a table.

    Args:
        client: Initialized Supabase client
        table_name: Name of the table to check policies for

    Returns:
        True if required policies are in place, False otherwise
    """
    try:
        # Query the information_schema.policies to check table policies
        response = (
            client.table("pg_policies")
            .select("policyname,cmd,qual,with_check")
            .eq("tablename", table_name)
            .execute()
        )

        if not response.data:
            logger.error("❌ No RLS policies found for %s table", table_name)
            return False

        logger.info("Found %d policies for %s table:", len(response.data), table_name)
        for policy in response.data:
            logger.info("  - %s (%s)", policy["policyname"], policy["cmd"])

        # Check for insert permission
        has_insert_policy = False
        for policy in response.data:
            # Check for permissive INSERT policy
            cmd = policy.get("cmd", "").upper()
            with_check = policy.get("with_check", "")

            if cmd == "INSERT" and "true" in str(with_check).lower():
                has_insert_policy = True
                logger.info(
                    "✅ Found permissive INSERT policy: %s", policy["policyname"]
                )
                break
            elif cmd == "ALL" and "true" in str(with_check).lower():
                has_insert_policy = True
                logger.info("✅ Found permissive ALL policy: %s", policy["policyname"])
                break

        if not has_insert_policy:
            logger.error(
                "❌ No permissive INSERT policy found for %s table", table_name
            )
            return False

        return True
    except Exception as e:
        logger.error("Error checking policies for %s table: %s", table_name, e)
        return False


def test_insert_permissions(client: Client, table_name: str) -> bool:
    """
    Test insert permissions by inserting a test record.

    Args:
        client: Initialized Supabase client
        table_name: Name of the table to test

    Returns:
        True if insert is successful, False otherwise
    """
    try:
        # Prepare test data
        test_data = {}
        if table_name == "request_logs":
            test_data = {
                "model": "test-model",
                "messages": json.dumps([{"role": "user", "content": "test"}]),
                "end_user": "test-validate-user",
                "status": "success",
                "response_time": 0.1,
                "litellm_call_id": f"test-validate-{time.time()}",
            }
        elif table_name == "usage_logs":
            test_data = {
                "user_id": "test-validate-user",
                "session_id": f"test-session-{time.time()}",
                "model_name": "test-model",
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            }

        # Insert test data
        logger.info("Testing insert permissions for %s table...", table_name)
        result = client.table(table_name).insert(test_data).execute()

        if result.data:
            logger.info(
                "✅ Successfully inserted test record into %s table", table_name
            )

            # Clean up by deleting the test record
            if table_name == "request_logs":
                client.table(table_name).delete().eq(
                    "litellm_call_id", test_data["litellm_call_id"]
                ).execute()
            elif table_name == "usage_logs":
                client.table(table_name).delete().eq(
                    "session_id", test_data["session_id"]
                ).execute()

            return True
        else:
            logger.error("❌ Failed to insert test record into %s table", table_name)
            if hasattr(result, "error"):
                logger.error("Error: %s", result.error)
            return False
    except Exception as e:
        logger.error("Error testing insert permissions for %s table: %s", table_name, e)
        return False


def main():
    """Main function to validate token tracking tables and policies."""
    # Get Supabase client
    client = get_supabase_client()
    if not client:
        logger.error("Failed to initialize Supabase client. Exiting.")
        return 1

    all_checks_passed = True

    # Check request_logs table
    if not check_table_exists(client, "request_logs"):
        all_checks_passed = False
    else:
        if not check_policies(client, "request_logs"):
            all_checks_passed = False

        # Import json and time for test insert
        import json
        import time

        if not test_insert_permissions(client, "request_logs"):
            all_checks_passed = False

    # Check usage_logs table
    if not check_table_exists(client, "usage_logs"):
        all_checks_passed = False
    else:
        if not check_policies(client, "usage_logs"):
            all_checks_passed = False

        # Import json and time for test insert if not already imported
        if "json" not in locals():
            import json
            import time

        if not test_insert_permissions(client, "usage_logs"):
            all_checks_passed = False

    # Summary
    if all_checks_passed:
        logger.info("\n✅ All checks passed! Token tracking should work correctly.")
        print("\nTo test token tracking, run:")
        print("python scripts/test_enhanced_litellm_callbacks.py --service-key")
        return 0
    else:
        logger.error("\n❌ Some checks failed. Token tracking may not work correctly.")
        print(
            "\nTo fix policies, you may need to run SQL commands in the Supabase SQL Editor to:"
        )
        print(
            "1. Create more permissive INSERT policies that allow ALL users to insert"
        )
        print("2. Ensure your service role has full access to both tables")
        print("\nExample policy for request_logs:")
        print(
            'CREATE POLICY "Allow all inserts to request_logs" ON public.request_logs FOR INSERT WITH CHECK (true);'
        )
        print("\nExample policy for usage_logs:")
        print(
            'CREATE POLICY "Allow all inserts to usage_logs" ON public.usage_logs FOR INSERT WITH CHECK (true);'
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
