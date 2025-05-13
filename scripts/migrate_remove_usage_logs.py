#!/usr/bin/env python3
"""
Script to remove the redundant usage_logs table from the database.

Since we've fully migrated to using the request_logs table for token usage tracking,
this script drops the legacy usage_logs table and any related triggers/functions.
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

# SQL to drop usage_logs table and related objects
DROP_USAGE_LOGS_SQL = """
-- Drop the trigger first
DROP TRIGGER IF EXISTS usage_logs_to_request_logs_trigger ON public.usage_logs;

-- Drop the function used by the trigger
DROP FUNCTION IF EXISTS migrate_usage_logs_to_request_logs();

-- Finally drop the table itself
DROP TABLE IF EXISTS public.usage_logs;
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
    """Run the SQL to drop the usage_logs table and related objects."""
    try:
        # First check if the table exists
        if not check_table_exists(supabase, "usage_logs"):
            logger.info("usage_logs table does not exist. No migration needed.")
            return True

        # Execute the migration SQL
        logger.info("Dropping usage_logs table and related objects...")
        supabase.sql(DROP_USAGE_LOGS_SQL).execute()

        # Verify the table was dropped
        if check_table_exists(supabase, "usage_logs"):
            logger.error("Failed to drop usage_logs table")
            return False

        logger.info("Successfully removed usage_logs table")
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
    logger.info("Starting migration to remove usage_logs table...")
    success = execute_migration(supabase)

    if success:
        logger.info("Migration completed successfully")
        sys.exit(0)
    else:
        logger.error("Migration failed")
        sys.exit(1)
