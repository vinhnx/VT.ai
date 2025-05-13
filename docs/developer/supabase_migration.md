# Supabase Migration Guide

This guide explains how to migrate remaining async Supabase calls in the VT.ai codebase to use the synchronous style with the `supabase_query` helper function.

## Overview

The VT.ai codebase is being migrated from an async Supabase client to a sync client. The main changes include:

1. Using the `supabase_query` helper function instead of directly using `supabase.table()`
2. Removing `await` keywords for Supabase operations
3. Maintaining consistent style for chaining query operations

## Progress Made

Key components that have been migrated:

- `get_user_data` function
- `track_usage` function
- `has_reached_limit` function
- Subscription plans endpoint
- Subscription status endpoint
- API key management endpoints
- Stripe integration endpoints

## Pending Migrations

Some functions still need to be updated:

### 1. Authentication with Supabase Auth

```python
# This needs to be updated:
user_response = await supabase.auth.get_user(token)

# Note: The auth module might still require async calls
```

### 2. LiteLLM Completion

```python
# The model completion will remain async:
response = await completion(
    model=request.model,
    messages=[message.model_dump() for message in request.messages],
    max_tokens=request.max_tokens,
)
```

### 3. Webhook Handlers

The following handlers need to be updated:

```python
await handle_subscription_event(event)
await handle_invoice_event(event)
await handle_checkout_completed(event)
```

## How to Update Functions

Follow these steps to update functions consistently:

1. Remove the `async` keyword from the function definition if all Supabase operations are sync
2. Replace `await supabase.table("table_name")` with `supabase_query("table_name")`
3. Use consistent syntax for query chains:

```python
# Updated format:
response = supabase_query("table_name") \
    .select("*") \
    .eq("column", value) \
    .execute()
```

4. Update function calls to remove `await` when calling updated functions:

```python
# Change from:
user_data = await get_user_data(user_id)

# To:
user_data = get_user_data(user_id)
```

## Testing Migrations

After updating functions:

1. Run the test script:

```bash
python scripts/test_supabase_refactor.py
```

2. Test the API endpoints directly:

```bash
curl -X GET "http://localhost:8000/api/subscription/plans" -H "accept: application/json"
```

3. Check for any runtime errors in the server logs.

## Handling Mixed Async/Sync Functions

Some functions may need to remain async if they call other async functions:

```python
async def chat(...):
    # Sync Supabase calls
    user_data = get_user_data(user_id)

    # Async LiteLLM call
    response = await completion(...)  # This is still async

    # Must remain async due to the await above
```

## Future Work

For a complete migration:

1. Update all functions in `/vtai/utils/auth_handler.py`
2. Update webhook handling in `/vtai/webhooks.py`
3. Update API key operations in `/vtai/api_keys.py`
4. Add proper error handling for sync operations
