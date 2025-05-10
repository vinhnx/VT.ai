#!/usr/bin/env python3
"""
Supabase Setup Script for VT.ai

This script creates the necessary database tables in Supabase for the VT.ai application.
Run this script once to set up your Supabase project for VT.ai.

Usage:
    python scripts/setup_supabase.py

Environment variables:
    SUPABASE_URL: Your Supabase project URL
    SUPABASE_KEY: Your Supabase service role key (not anon key)
"""

import asyncio
import os
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from supabase import Client, create_client

# Enable importing from the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables from .env file
load_dotenv()


# SQL to create usage_logs table
CREATE_USAGE_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS usage_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,
    session_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cost FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE SET NULL
);

-- Add indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_usage_logs_user_id ON usage_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_logs_session_id ON usage_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_usage_logs_created_at ON usage_logs(created_at);

-- Set up RLS (Row Level Security) for usage_logs
ALTER TABLE usage_logs ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only view their own logs
CREATE POLICY usage_logs_user_select ON usage_logs
    FOR SELECT USING (auth.uid() = user_id);

-- Policy: Allow anonymous logs (user_id is NULL)
CREATE POLICY usage_logs_anon_insert ON usage_logs
    FOR INSERT WITH CHECK (true);

-- Policy: Backend service can read all logs
CREATE POLICY usage_logs_service_select ON usage_logs
    FOR SELECT USING (auth.uid() IN (
        SELECT id FROM auth.users WHERE email = 'service@example.com'
    ));
"""

# SQL to create user_subscriptions table (for Phase 2)
CREATE_USER_SUBSCRIPTIONS_TABLE = """
-- This table will be implemented in Phase 2 for subscription management
CREATE TABLE IF NOT EXISTS user_subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    subscription_tier TEXT NOT NULL DEFAULT 'free',
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    token_limit INTEGER,
    next_billing_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_user_subscriptions_user_id ON user_subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_subscriptions_is_active ON user_subscriptions(is_active);

-- Set up RLS
ALTER TABLE user_subscriptions ENABLE ROW LEVEL SECURITY;

-- Policy: Users can view their own subscription
CREATE POLICY user_subscriptions_user_select ON user_subscriptions
    FOR SELECT USING (auth.uid() = user_id);

-- Policy: Service role can manage subscriptions
CREATE POLICY user_subscriptions_service ON user_subscriptions
    USING (auth.uid() IN (
        SELECT id FROM auth.users WHERE email = 'service@example.com'
    ));
"""


def execute_sql_query(supabase: Client, query: str) -> Dict[str, Any]:
    """
    Execute a SQL query using Supabase.

    Args:
        supabase: The Supabase client
        query: The SQL query to execute

    Returns:
        The result of the query
    """
    try:
        # Use synchronous version for simplicity
        result = supabase.rpc("pgclient", {"query": query}).execute()
        return result
    except Exception as e:
        print(f"Error executing SQL: {e}")
        raise


def setup_supabase():
    """
    Set up the necessary database tables and RLS policies in Supabase.
    """
    # Get Supabase credentials from environment
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
        print("Please set these variables in your .env file or environment.")
        sys.exit(1)

    try:
        # Initialize Supabase client
        supabase = create_client(supabase_url, supabase_key)
        print("Connected to Supabase successfully.")

        # Create the usage_logs table
        print("Creating usage_logs table...")
        execute_sql_query(supabase, CREATE_USAGE_LOGS_TABLE)
        print("✅ usage_logs table created successfully.")

        # Create the user_subscriptions table for Phase 2
        print("Creating user_subscriptions table (for Phase 2)...")
        execute_sql_query(supabase, CREATE_USER_SUBSCRIPTIONS_TABLE)
        print("✅ user_subscriptions table created successfully.")

        print(
            "\nDatabase setup complete! Your Supabase project is now ready for VT.ai."
        )
        print("\nNext steps:")
        print(
            "1. Make sure your OAuth providers are configured in the Supabase dashboard"
        )
        print("2. Update your .env file with the necessary Supabase credentials")
        print("3. Start the VT.ai application")

    except Exception as e:
        print(f"Error setting up Supabase: {e}")
        sys.exit(1)


if __name__ == "__main__":
    setup_supabase()
