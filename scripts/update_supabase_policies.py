#!/usr/bin/env python3
"""
Updates Supabase RLS policies to enable password-authenticated users to log usage.

This script modifies the row-level security policies for the usage_logs table
to ensure all users can insert records, including those using password authentication.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

# Load environment variables from .env file
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# SQL to update RLS policy for usage_logs
UPDATE_USAGE_LOGS_POLICY = """
-- Drop existing policy
DROP POLICY IF EXISTS usage_logs_anon_insert ON usage_logs;

-- Create a more permissive insert policy
CREATE POLICY usage_logs_anon_insert ON usage_logs
    FOR INSERT WITH CHECK (true);

-- Create policy for all users to insert logs
CREATE POLICY IF NOT EXISTS usage_logs_all_insert ON usage_logs
    FOR INSERT WITH CHECK (true);

-- Add a password_auth column to users table if it doesn't exist
ALTER TABLE IF EXISTS users
    ADD COLUMN IF NOT EXISTS password_auth BOOLEAN DEFAULT false;

-- Add a username column to users table if it doesn't exist
ALTER TABLE IF EXISTS users
    ADD COLUMN IF NOT EXISTS username TEXT UNIQUE;

-- Modify the user_id column in usage_logs to accept text instead of UUID
ALTER TABLE usage_logs
    ALTER COLUMN user_id TYPE TEXT USING user_id::TEXT;

-- Add auth_method column to usage_logs if it doesn't exist
ALTER TABLE usage_logs
    ADD COLUMN IF NOT EXISTS auth_method TEXT DEFAULT 'supabase';
"""


def execute_sql_query(supabase: Client, query: str) -> None:
    """
    Execute a SQL query via Supabase.

    Args:
        supabase: The Supabase client
        query: The SQL query to execute
    """
    try:
        # Since rpc method may not be available in all Supabase client versions,
        # we'll print the SQL statements that would be executed and manually
        # apply them through the Supabase dashboard

        print(f"\nSQL to execute via Supabase dashboard:\n{query}")
        print("\nPlease apply these SQL statements through the Supabase SQL Editor.")
        print("You can copy the SQL above and run it in the Supabase SQL Editor.")

        # For more advanced applications, consider using Supabase REST API
        # or PostgreSQL client libraries directly

    except Exception as e:
        print(f"Error processing SQL: {e}")
        raise


def update_rls_policies(supabase: Client) -> None:
    """
    Update the row-level security policies for the usage_logs table.

    Args:
        supabase: The Supabase client
    """
    print("Updating RLS policies for usage_logs table...")
    execute_sql_query(supabase, UPDATE_USAGE_LOGS_POLICY)
    print("SQL for RLS policy updates displayed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update Supabase RLS policies")
    parser.add_argument(
        "--check", action="store_true", help="Check RLS policies but don't update"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Show SQL to execute manually in Supabase dashboard",
    )

    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL or SUPABASE_KEY not found in environment variables.")
        sys.exit(1)

    print(f"URL: {SUPABASE_URL[:10]}...")
    print(f"Key present: {bool(SUPABASE_KEY)}")

    try:
        # Create Supabase client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client created successfully")

        if args.check:
            print("Check mode: Would update RLS policies but not actually doing it.")
            print("Run without --check to apply changes.")
        elif args.manual:
            # Display SQL to execute manually
            print("\nSQL statements to execute in the Supabase SQL Editor:")
            print(UPDATE_USAGE_LOGS_POLICY)
        else:
            update_rls_policies(supabase)
            print("Instructions for updating RLS policies completed.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
