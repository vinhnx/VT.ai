# Supabase Integration Guide for VT.ai

This guide explains how to set up and configure the Supabase integration for VT.ai's subscription service.

## Prerequisites

- A Supabase account and project
- Supabase project URL and API keys (anon key and service role key)
- Python 3.x installed
- VT.ai codebase cloned locally

## Environment Setup

1. Create a `.env` file in the project root (or copy the `.env.example` file):

```
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-key                   # For regular API operations
SUPABASE_SERVICE_KEY=your-service-role-key   # For schema operations

# Stripe Configuration (for subscription management)
STRIPE_SECRET_KEY=your-stripe-secret-key
STRIPE_WEBHOOK_SECRET=your-webhook-secret
```

## Supabase Database Setup

The VT.ai subscription service requires several tables in your Supabase database. Due to limitations in the Supabase Python client, you'll need to set up the schema manually:

1. Validate your Supabase connection:

```bash
python scripts/validate_supabase.py
```

2. Run the setup script to get instructions:

```bash
python scripts/setup_supabase.py
```

3. Follow the displayed instructions to create the required database schema:
   - Log in to your Supabase dashboard
   - Go to the SQL Editor
   - Create a new query
   - Copy the contents of `scripts/supabase_schema.sql`
   - Execute the query

## Database Tables

The schema creates the following tables:

1. **user_profiles**: Stores subscription information and usage statistics for each user
2. **request_logs**: Records detailed token usage for analytics and auditing
3. **stripe_customers**: Maps Supabase users to Stripe customers
4. **subscription_plans**: Defines the available subscription tiers and their limits
5. **api_keys**: Manages API keys for authentication as an alternative to JWT tokens

## Row-Level Security (RLS) Policies

Supabase uses Row-Level Security (RLS) to control access to data. The schema includes the following security policies:

- Users can only view and update their own data
- Anyone can view subscription plans
- Users can manage their own API keys

## Usage Tracking

To ensure proper token usage tracking, you may need to update the RLS policies for the usage_logs table:

```bash
python scripts/update_supabase_policies.py --manual
```

This will display SQL statements you should execute in the Supabase SQL Editor to update the policies.

## Schema Verification

To verify that your schema is correctly set up and has the required columns:

```bash
python scripts/check_supabase_schema.py
```

This will check if all the necessary tables and columns exist and suggest fixes for any issues.

## Subscription Plans

The schema automatically inserts three subscription plans:

1. **Free**: 10,000 tokens limit, $0.00
2. **Basic**: 100,000 tokens limit, $15.00
3. **Premium**: 500,000 tokens limit, $49.00

You'll need to create corresponding products and prices in your Stripe dashboard.

## API Integration

The VT.ai server integrates with Supabase for user authentication and token usage tracking. When users make API requests, the system:

1. Authenticates users via JWT token or API key
2. Checks their subscription tier and token limits
3. Logs token usage for billing and analytics
4. Enforces rate limits based on subscription tier

### Supabase Client Implementation

The application uses a sync-style Supabase client (non-async). To ensure uniform access to tables, we use the `supabase_query` helper function:

```python
def supabase_query(table_name: str):
    """
    Returns a query builder for the specified table, compatible with both
    sync and async versions of the Supabase client.
    """
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable.",
        )
    return supabase.table(table_name)
```

This function abstracts the Supabase client implementation and allows for consistent table access patterns:

```python
# Example of querying user profile data
response = supabase_query("user_profiles") \
    .select("*") \
    .eq("user_id", user_id) \
    .maybe_single() \
    .execute()
```

## Troubleshooting

If you encounter issues with the Supabase integration:

1. Check that your Supabase URL and API keys are correct
2. Verify that all required tables exist in your database
3. Ensure that RLS policies are properly configured
4. Look for error messages in the server logs

For advanced troubleshooting, use:

```bash
python scripts/check_supabase_schema.py --fix
```

This will suggest SQL fixes for common schema issues.

## Testing Supabase Integration

To verify that the Supabase integration is working correctly after code changes:

```bash
python scripts/test_supabase_refactor.py
```

This script tests:

- Connection to all tables in the database
- Basic CRUD operations (select, insert, update, delete)
- User profile management
- API key creation and validation

You can set a test user ID in your environment for more thorough testing:

```bash
export TEST_USER_ID=your-test-user-id
python scripts/test_supabase_refactor.py
```

### Testing the API Endpoints

To test the API endpoints after making changes:

1. Start the API server:

```bash
python vtai/main.py
```

2. Test the subscription plans endpoint:

```bash
curl -X GET "http://localhost:8000/api/subscription/plans" -H "accept: application/json"
```

3. Test user-specific endpoints (requires authentication):

```bash
curl -X GET "http://localhost:8000/api/subscription/status" \
  -H "accept: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```
