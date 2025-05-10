#!/usr/bin/env python3
"""
Diagnostics script for checking Supabase database schema for VT.ai.

This script checks if the Supabase database tables and columns are properly configured
for token usage logging with both authentication methods.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv


def format_table_info(table_info: List[Dict]) -> str:
    """Format table information for display."""
    if not table_info:
        return "No table information available."

    result = []
    for column in table_info:
        result.append(f"  • {column.get('column_name')} ({column.get('data_type')})")
        if column.get("is_nullable") == "NO":
            result[-1] += " [NOT NULL]"

    return "\n".join(result)


def main():
    parser = argparse.ArgumentParser(
        description="Check Supabase database schema for VT.ai"
    )
    parser.add_argument("--fix", action="store_true", help="Generate SQL to fix issues")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Check if Supabase credentials are available
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ Supabase credentials not found in environment variables.")
        print("   Make sure SUPABASE_URL and SUPABASE_KEY are set in .env file.")
        sys.exit(1)

    try:
        # Import supabase after checking environment variables
        from supabase import Client, create_client

        # Connect to Supabase
        print(f"📡 Connecting to Supabase at {supabase_url[:16]}...")
        supabase = create_client(supabase_url, supabase_key)

        # Check if usage_logs table exists and get its structure
        print("\n📊 Checking database schema...")

        # Check usage_logs table
        usage_logs_info = None
        try:
            usage_logs_info = (
                supabase.table("information_schema.columns")
                .select("column_name, data_type, is_nullable")
                .eq("table_name", "usage_logs")
                .execute()
            )

            if usage_logs_info.data:
                print("✅ usage_logs table exists with the following columns:")
                print(format_table_info(usage_logs_info.data))

                # Check user_id column type
                user_id_column = next(
                    (
                        col
                        for col in usage_logs_info.data
                        if col.get("column_name") == "user_id"
                    ),
                    None,
                )
                if user_id_column:
                    if user_id_column.get("data_type") == "uuid":
                        print(
                            "❌ user_id column is UUID type, but should be TEXT for compatibility"
                        )
                        if args.fix:
                            print(
                                "\nTo fix this issue, run the following SQL in Supabase SQL Editor:"
                            )
                            print(
                                "ALTER TABLE usage_logs ALTER COLUMN user_id TYPE TEXT USING user_id::TEXT;"
                            )
                    elif user_id_column.get("data_type") == "text":
                        print("✅ user_id column is correctly set to TEXT type")
                    else:
                        print(
                            f"⚠️ user_id column has unexpected type: {user_id_column.get('data_type')}"
                        )
                else:
                    print("❌ user_id column not found in usage_logs table")

                # Check if auth_method column exists
                auth_method_column = next(
                    (
                        col
                        for col in usage_logs_info.data
                        if col.get("column_name") == "auth_method"
                    ),
                    None,
                )
                if not auth_method_column:
                    print("⚠️ auth_method column does not exist in usage_logs table")
                    if args.fix:
                        print(
                            "\nTo fix this issue, run the following SQL in Supabase SQL Editor:"
                        )
                        print(
                            "ALTER TABLE usage_logs ADD COLUMN auth_method TEXT DEFAULT 'supabase';"
                        )
                else:
                    print("✅ auth_method column exists in usage_logs table")
            else:
                print("❌ usage_logs table not found")
                if args.fix:
                    print(
                        "\nTo create the usage_logs table, run setup_supabase.py script"
                    )
        except Exception as e:
            print(f"❌ Error checking usage_logs table: {e}")

        # Check users table
        users_info = None
        try:
            users_info = (
                supabase.table("information_schema.columns")
                .select("column_name, data_type, is_nullable")
                .eq("table_name", "users")
                .execute()
            )

            if users_info.data:
                print("\n✅ users table exists with the following columns:")
                print(format_table_info(users_info.data))

                # Check if username column exists
                username_column = next(
                    (
                        col
                        for col in users_info.data
                        if col.get("column_name") == "username"
                    ),
                    None,
                )
                if not username_column:
                    print("⚠️ username column does not exist in users table")
                    if args.fix:
                        print(
                            "\nTo fix this issue, run the following SQL in Supabase SQL Editor:"
                        )
                        print("ALTER TABLE users ADD COLUMN username TEXT UNIQUE;")
                else:
                    print("✅ username column exists in users table")

                # Check if password_auth column exists
                password_auth_column = next(
                    (
                        col
                        for col in users_info.data
                        if col.get("column_name") == "password_auth"
                    ),
                    None,
                )
                if not password_auth_column:
                    print("⚠️ password_auth column does not exist in users table")
                    if args.fix:
                        print(
                            "\nTo fix this issue, run the following SQL in Supabase SQL Editor:"
                        )
                        print(
                            "ALTER TABLE users ADD COLUMN password_auth BOOLEAN DEFAULT false;"
                        )
                else:
                    print("✅ password_auth column exists in users table")
            else:
                print("❌ users table not found")
        except Exception as e:
            print(f"❌ Error checking users table: {e}")

        # Check RLS policies
        try:
            rls_policies = (
                supabase.table("pg_policies")
                .select("tablename, policyname, cmd, qual")
                .execute()
            )

            if rls_policies.data:
                print("\n📋 Row-Level Security (RLS) Policies:")
                usage_log_policies = [
                    p for p in rls_policies.data if p.get("tablename") == "usage_logs"
                ]

                if usage_log_policies:
                    for policy in usage_log_policies:
                        print(
                            f"  • {policy.get('policyname')} ({policy.get('cmd')}) - {policy.get('qual')}"
                        )

                    # Check for permissive insert policy
                    permissive_insert = any(
                        p.get("cmd") == "INSERT"
                        and (p.get("qual") == "true" or "true" in str(p.get("qual")))
                        for p in usage_log_policies
                    )

                    if permissive_insert:
                        print("✅ Permissive INSERT policy found for usage_logs table")
                    else:
                        print(
                            "❌ No permissive INSERT policy found for usage_logs table"
                        )
                        if args.fix:
                            print(
                                "\nTo fix this issue, run the following SQL in Supabase SQL Editor:"
                            )
                            print(
                                "CREATE POLICY usage_logs_all_insert ON usage_logs FOR INSERT WITH CHECK (true);"
                            )
                else:
                    print("❌ No RLS policies found for usage_logs table")
                    if args.fix:
                        print(
                            "\nTo fix this issue, run the following SQL in Supabase SQL Editor:"
                        )
                        print(
                            "CREATE POLICY usage_logs_all_insert ON usage_logs FOR INSERT WITH CHECK (true);"
                        )
            else:
                print("⚠️ Unable to retrieve RLS policies information")
        except Exception as e:
            print(f"❌ Error checking RLS policies: {e}")

        # Summary
        print("\n🔍 Summary:")
        if args.fix:
            print(
                "Run the SQL commands shown above in the Supabase SQL Editor to fix any issues."
            )
            print(
                "Alternatively, run the get_supabase_policy_sql.py script to get all the necessary SQL at once."
            )
        else:
            print("Run this script with --fix to see SQL commands to fix any issues.")

    except ImportError:
        print("❌ Supabase client not installed.")
        print("   Run: uv pip install -U supabase python-dotenv")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    main()
