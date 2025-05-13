#!/usr/bin/env python
"""
Update Supabase RLS policies for token usage tracking tables.

This script updates the row-level security policies for request_logs and usage_logs tables
to ensure that the token usage tracking system works correctly.
"""

import argparse
import logging
import os
import sys
from typing import Optional

import dotenv
from supabase import Client, create_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vt.ai.setup")

# Load environment variables
dotenv.load_dotenv()


def get_supabase_client() -> Optional[Client]:
    """
    Initialize the Supabase client using environment variables.

    Returns:
        Initialized Supabase client or None if initialization fails
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    # For RLS policy updates, we must use the service key
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not supabase_url or not supabase_key:
        logger.error(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables."
        )
        return None

    try:
        # Initialize with service key to have admin privileges
        client = create_client(supabase_url, supabase_key)
        logger.info("Supabase admin client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return None


def update_rls_policies(client: Client) -> bool:
    """
    Update row-level security policies for request_logs and usage_logs tables.

    Args:
        client: Initialized Supabase client

    Returns:
        True if the policies were updated successfully, False otherwise
    """
    # SQL to update RLS policies
    update_sql = """
    -- For request_logs table
    -- First drop existing policies
    DROP POLICY IF EXISTS "Users can insert their own request logs" ON public.request_logs;
    DROP POLICY IF EXISTS "Anonymous users can insert request logs" ON public.request_logs;

    -- Create more permissive insert policies
    CREATE POLICY "Allow all inserts to request_logs"
    ON public.request_logs
    FOR INSERT WITH CHECK (true);

    -- Create policy for service role to do anything
    CREATE POLICY "Service role has full access to request_logs"
    ON public.request_logs
    USING (auth.role() = 'service_role');

    -- Keep the select policy as is - users should only see their own logs
    CREATE POLICY IF NOT EXISTS "Users can view their own request logs"
    ON public.request_logs
    FOR SELECT
    USING (auth.uid()::TEXT = end_user OR (auth.role() = 'service_role'));

    -- For usage_logs table
    -- First drop existing policies
    DROP POLICY IF EXISTS "Users can insert their own usage logs" ON public.usage_logs;
    DROP POLICY IF EXISTS "Anonymous users can insert usage logs" ON public.usage_logs;

    -- Create more permissive insert policies
    CREATE POLICY "Allow all inserts to usage_logs"
    ON public.usage_logs
    FOR INSERT WITH CHECK (true);

    -- Create policy for service role to do anything
    CREATE POLICY "Service role has full access to usage_logs"
    ON public.usage_logs
    USING (auth.role() = 'service_role');

    -- Keep the select policy as is - users should only see their own logs
    CREATE POLICY IF NOT EXISTS "Users can view their own usage logs"
    ON public.usage_logs
    FOR SELECT
    USING (auth.uid()::TEXT = user_id OR (auth.role() = 'service_role'));
    """

    try:
        logger.info("Updating RLS policies...")
        # Use the REST API to execute SQL (postgrest.rpc)
        result = client.rpc("exec_sql", {"sql": update_sql}).execute()
        logger.info("Successfully updated RLS policies.")
        return True
    except Exception as e:
        logger.error(f"Error updating RLS policies: {e}")
        return False


def verify_policies(client: Client) -> bool:
    """
    Verify that the RLS policies have been correctly applied.

    Args:
        client: Initialized Supabase client

    Returns:
        True if the policies are correctly applied, False otherwise
    """
    try:
        # Query the information_schema.policies to check if our policies exist
        response = (
            client.table("information_schema.policies")
            .select("tablename,policyname,permissive,cmd,qual,with_check")
            .in_("tablename", ["request_logs", "usage_logs"])
            .execute()
        )

        if not response.data:
            logger.error("No RLS policies found for request_logs or usage_logs tables.")
            return False

        logger.info(f"Found {len(response.data)} RLS policies:")
        for policy in response.data:
            logger.info(
                f"  {policy['tablename']}: {policy['policyname']} - {policy['cmd']}"
            )

        # Check for the specific policies we need
        required_policies = [
            ("request_logs", "Allow all inserts to request_logs"),
            ("request_logs", "Service role has full access to request_logs"),
            ("usage_logs", "Allow all inserts to usage_logs"),
            ("usage_logs", "Service role has full access to usage_logs"),
        ]

        missing_policies = []
        for table, policy_name in required_policies:
            found = False
            for policy in response.data:
                if policy["tablename"] == table and policy["policyname"] == policy_name:
                    found = True
                    break
            if not found:
                missing_policies.append(f"{table}: {policy_name}")

        if missing_policies:
            logger.error("Missing required policies:")
            for policy in missing_policies:
                logger.error(f"  {policy}")
            return False

        logger.info("All required RLS policies are in place.")
        return True
    except Exception as e:
        logger.error(f"Error verifying RLS policies: {e}")
        return False


def main():
    """Main function to update and verify RLS policies."""
    parser = argparse.ArgumentParser(
        description="Update Supabase RLS policies for token usage tracking."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing policies without updating them.",
    )
    args = parser.parse_args()

    client = get_supabase_client()
    if not client:
        logger.error("Failed to initialize Supabase client. Exiting.")
        sys.exit(1)

    if args.verify_only:
        logger.info("Verifying existing RLS policies...")
        if verify_policies(client):
            logger.info("RLS policies verification successful.")
            sys.exit(0)
        else:
            logger.error("RLS policies verification failed.")
            sys.exit(1)
    else:
        logger.info("Updating RLS policies...")
        if update_rls_policies(client):
            logger.info("RLS policies updated successfully.")

            logger.info("Verifying updated RLS policies...")
            if verify_policies(client):
                logger.info("RLS policies verification successful.")
                sys.exit(0)
            else:
                logger.error("RLS policies verification failed after update.")
                sys.exit(1)
        else:
            logger.error("Failed to update RLS policies.")
            sys.exit(1)


if __name__ == "__main__":
    main()
