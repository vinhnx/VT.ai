#!/usr/bin/env python3
"""
Script to update RLS (Row Level Security) policies for the request_logs table.

This script ensures that proper RLS policies are in place for the request_logs table,
allowing authenticated users to insert logs and view their own logs.
"""

import logging
import os
import sys
from typing import Optional

import dotenv
from supabase import create_client

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# SQL to set up RLS policies for the request_logs table
UPDATE_REQUEST_LOGS_POLICY = """
-- Enable RLS on request_logs table if not already enabled
ALTER TABLE public.request_logs ENABLE ROW LEVEL SECURITY;

-- Drop existing policies to avoid duplicates
DROP POLICY IF EXISTS "Allow authenticated users to insert request logs" ON public.request_logs;
DROP POLICY IF EXISTS "Allow users to view their own request logs" ON public.request_logs;
DROP POLICY IF EXISTS "Allow service role to view all request logs" ON public.request_logs;

-- Create policy for inserting logs (authenticated users)
CREATE POLICY "Allow authenticated users to insert request logs"
ON public.request_logs
FOR INSERT
TO authenticated
WITH CHECK (true);

-- Create policy for selecting logs (users can see their own logs)
CREATE POLICY "Allow users to view their own request logs"
ON public.request_logs
FOR SELECT
TO authenticated
USING (end_user = auth.uid());

-- Create policy for service role (can see all logs)
CREATE POLICY "Allow service role to view all request logs"
ON public.request_logs
FOR ALL
TO service_role
USING (true);
"""


def connect_to_supabase() -> Optional[object]:
    """Connect to Supabase using environment variables."""
    try:
        # Load environment variables
        dotenv.load_dotenv()

        # Get Supabase credentials
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            logger.error("Missing Supabase credentials in environment variables")
            return None

        # Create Supabase client
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        logger.error(f"Error connecting to Supabase: {e}")
        return None


def check_table_exists(supabase, table_name: str) -> bool:
    """Check if a table exists in the database."""
    try:
        # Query the information_schema to check if the table exists
        response = (
            supabase.table("information_schema.tables")
            .select("table_name")
            .eq("table_schema", "public")
            .eq("table_name", table_name)
            .execute()
        )

        exists = len(response.data) > 0
        logger.info(f"Table '{table_name}' exists: {exists}")
        return exists
    except Exception as e:
        logger.error(f"Error checking if table '{table_name}' exists: {e}")
        return False


def execute_migration(supabase) -> bool:
    """Run the SQL to update RLS policies for the request_logs table."""
    try:
        # First check if the table exists
        if not check_table_exists(supabase, "request_logs"):
            logger.error("request_logs table does not exist. Please create it first.")
            return False

        # Execute the migration SQL
        logger.info("Updating RLS policies for request_logs table...")
        supabase.sql(UPDATE_REQUEST_LOGS_POLICY).execute()

        logger.info("Successfully updated RLS policies for request_logs table")
        return True
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return False


if __name__ == "__main__":
    # Connect to Supabase
    logger.info("Connecting to Supabase...")
    supabase = connect_to_supabase()

    if not supabase:
        logger.error("Failed to connect to Supabase. Exiting.")
        sys.exit(1)

    # Run the migration
    logger.info("Starting migration to update request_logs RLS policies...")
    success = execute_migration(supabase)

    if success:
        logger.info("Migration completed successfully")
        sys.exit(0)
    else:
        logger.error("Migration failed")
        sys.exit(1)
