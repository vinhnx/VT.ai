#!/usr/bin/env python3
"""
Supabase Setup Script for VT.ai

This script validates the connection to Supabase and provides instructions
for setting up the database schema for the VT.ai application.

Usage:
    python scripts/setup_supabase.py

Environment variables:
    SUPABASE_URL: Your Supabase project URL
    SUPABASE_SERVICE_KEY: Your Supabase service role key (not anon key)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

# Load environment variables from .env file
load_dotenv()


def setup_supabase():
    """
    Check connection to Supabase and provide instructions for schema setup.
    """
    # Get Supabase credentials from environment
    supabase_url = os.getenv("SUPABASE_URL")
    # For schema operations, we need the service role key
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_KEY"))

    if not supabase_url or not supabase_key:
        print(
            "Error: SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables must be set."
        )
        print("Please set these variables in your .env file or environment.")
        print(
            "NOTE: For schema operations, you need to use the service role key, not the anon key."
        )
        sys.exit(1)

    try:
        # Initialize Supabase client
        supabase = create_client(supabase_url, supabase_key)
        print("Connected to Supabase successfully.")

        # Get the path to the SQL schema file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(script_dir, "supabase_schema.sql")

        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                schema_sql = f.read()
                print(f"Loaded schema SQL from {schema_path}")
        else:
            print(f"❌ Schema file not found at {schema_path}")
            sys.exit(1)

        print("\n===== IMPORTANT: MANUAL SQL EXECUTION REQUIRED =====")
        print(
            "The Supabase Python client doesn't support direct SQL execution for DDL statements."
        )
        print("Please follow these steps to set up your database schema:")
        print("\n1. Log in to your Supabase Dashboard")
        print("2. Go to the SQL Editor")
        print("3. Create a new query")
        print(f"4. Copy the contents of {schema_path} into the editor")
        print("5. Execute the query")
        print("\n=================================================")

        # Try to validate if some tables already exist
        tables_to_check = [
            "user_profiles",
            "request_logs",
            "stripe_customers",
            "subscription_plans",
            "api_keys",
        ]

        print("\nChecking for existing tables...")
        for table in tables_to_check:
            try:
                response = supabase.table(table).select("*").limit(1).execute()
                print(f"✅ Table '{table}' exists!")
            except Exception as e:
                print(f"❓ Table '{table}' may not exist yet: {e}")

        print("\nDatabase connection validated!")
        print("\nNext steps:")
        print("1. Make sure you execute the SQL schema in the Supabase SQL Editor")
        print("2. Set up your OAuth providers in the Supabase dashboard")
        print(
            "3. Configure Stripe with the following products and price IDs: price_free, price_basic, price_premium"
        )
        print("4. Update your .env file with all necessary credentials")
        print("5. Start the VT.ai API server with 'python vtai/main.py'")

    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        sys.exit(1)


if __name__ == "__main__":
    setup_supabase()
