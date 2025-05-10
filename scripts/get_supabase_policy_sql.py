#!/usr/bin/env python3
"""
Quick script to display the SQL needed to update Supabase RLS policies.

The SQL output from this script should be executed in the Supabase SQL Editor
to update the row-level security policies for the usage_logs table.
"""

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

print("SQL to execute in the Supabase SQL Editor:")
print("=" * 50)
print(UPDATE_USAGE_LOGS_POLICY)
print("=" * 50)
print("\nCopy the SQL above and paste it into the Supabase SQL Editor.")
print(
    "This will update the RLS policies to allow password-authenticated users to log usage."
)
