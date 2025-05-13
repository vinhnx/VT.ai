# Token Usage Monitoring

## Overview

This document describes the token usage monitoring system integrated with LiteLLM callbacks in the VT.ai platform. The system tracks token usage across various LLM providers and stores the data in Supabase for monitoring and subscription tier management.

## Architecture

The token usage monitoring system consists of the following components:

1. **LiteLLM Callbacks**: A custom callback handler that extends LiteLLM's SupabaseHandler to add user identification and track token usage.
2. **Usage Logging Tables**: Two tables in Supabase store token usage data:
   - `request_logs`: Detailed logs of each LLM request, including token usage, model, and user identification
   - `usage_logs`: Summary of token usage for each user, used for subscription tier management

3. **Subscription Tier Management**: Token usage data is used to enforce subscription tier limits in the API endpoints.

## Implementation Details

### LiteLLM Callbacks

The custom callback handler is implemented in `vtai/utils/litellm_callbacks.py`. It extends the LiteLLM SupabaseHandler to add user identification and logging to both `request_logs` and `usage_logs` tables.

```python
from litellm.integrations.supabase import SupabaseHandler

class VTAISupabaseHandler(SupabaseHandler):
    """
    Custom Supabase callback handler for VT.ai token usage tracking.
    Extends LiteLLM's SupabaseHandler to add user identification.
    """

    def __init__(
        self,
        supabase_client: Client,
        table_name: str = "request_logs",
        log_to_legacy: bool = True,
    ):
        """
        Initialize the VT.ai Supabase callback handler.

        Args:
            supabase_client: The initialized Supabase client
            table_name: The Supabase table name to log to
            log_to_legacy: Whether to also log to the legacy usage_logs table
        """
        # Call parent init
        super().__init__(supabase_client=supabase_client, table_name=table_name)

        # Store additional properties
        self.client = supabase_client
        self.table_name = table_name
        self.log_to_legacy = log_to_legacy
```

### Usage Logging Tables

Two tables in Supabase store token usage data:

- `request_logs`: Created by `scripts/create_request_logs_table.py`, stores detailed logs of each LLM request.
- `usage_logs`: A legacy table that's also updated for backwards compatibility.

### Initialization

The LiteLLM callbacks are initialized in `vtai/utils/config.py` during application startup:

```python
def initialize_app(
    supabase_client: Optional[Client] = None,
    log_to_legacy: bool = True,
) -> Tuple[RouteLayer, None, OpenAI, AsyncOpenAI]:
    # ...

    # Initialize LiteLLM callbacks for token usage tracking
    try:
        from vtai.utils.litellm_callbacks import initialize_litellm_callbacks
        initialize_litellm_callbacks(supabase_client, log_to_legacy=log_to_legacy)
        logger.info("Initialized LiteLLM callbacks for token usage tracking")
    except Exception as e:
        logger.error(f"Failed to initialize LiteLLM callbacks: {e}")

    # ...
```

The `log_to_legacy` parameter controls whether token usage is also logged to the legacy `usage_logs` table. This parameter is set to `True` by default to maintain backward compatibility, but can be set to `False` to avoid duplicate logging once the migration is complete.

## Usage

The token usage monitoring system works automatically in the background. When an LLM request is made using LiteLLM, the callbacks are triggered and token usage is recorded in Supabase.

Example API usage:

```python
from litellm import completion

# This will automatically trigger the token usage callbacks
response = await completion(
    model="openrouter/qwen/qwen3-0.6b-04-28:free",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
```

## Subscription Tier Management

Token usage data is used to enforce subscription tier limits in the API endpoints. The limits are defined in the `has_reached_limit` function in `vtai/main.py`:

```python
# Define tier-based token limits
tier_limits = {
    "free": 10000,      # 10k tokens for free tier
    "basic": 100000,    # 100k tokens for basic tier
    "premium": 500000,  # 500k tokens for premium tier
    "unlimited": float("inf"),  # No limit for unlimited tier
}
```

## Future Improvements

1. **Cost Calculation**: Implement cost calculation based on token usage and model pricing.
2. **User Dashboard**: Create a dashboard for users to view their token usage and subscription status.
3. **Real-time Monitoring**: Implement real-time monitoring of token usage.
4. **Automatic Tier Upgrades**: Implement automatic tier upgrades when a user reaches their limit.

## Data Migration

The system includes tools for migrating data from the legacy `usage_logs` table to the new `request_logs` table:

### Batch Migration

The `scripts/migrate_usage_logs.py` script performs batch migration of historical data:

```python
def migrate_usage_logs():
    """Migrates data from usage_logs to request_logs."""
    # Processing in batches
    batch_size = 100
    offset = 0

    while offset < total_records:
        # Get batch from usage_logs
        records = supabase.table("usage_logs")
            .select("*")
            .range(offset, offset + batch_size - 1)
            .execute()
            .data

        # Transform and insert into request_logs
        # ...
```

### Continuous Migration

A Postgres trigger automatically migrates new entries from `usage_logs` to `request_logs`:

```sql
CREATE OR REPLACE FUNCTION public.migrate_usage_log_to_request_log()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.request_logs (
        model,
        messages,
        end_user,
        status,
        response_time,
        total_cost,
        additional_details
    )
    VALUES (
        NEW.model_name,
        '[]'::json,
        NEW.user_id,
        'success',
        0,
        NEW.cost,
        json_build_object(
            'session_id', NEW.session_id,
            'input_tokens', NEW.input_tokens,
            'output_tokens', NEW.output_tokens,
            'total_tokens', NEW.total_tokens,
            'auth_method', NEW.auth_method,
            'migrated_from_usage_logs', true,
            'migrated_at', now(),
            'usage_log_id', NEW.id
        )
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER usage_logs_to_request_logs_trigger
AFTER INSERT ON public.usage_logs
FOR EACH ROW
EXECUTE FUNCTION public.migrate_usage_log_to_request_log();
```

## Row-Level Security

The `request_logs` table is protected by Row-Level Security (RLS) policies to ensure users can only access their own data:

```sql
-- Enable RLS on request_logs table
ALTER TABLE public.request_logs ENABLE ROW LEVEL SECURITY;

-- Create policy for users to view their own request logs
CREATE POLICY "Users can view their own request logs"
ON public.request_logs
FOR SELECT
USING (auth.uid()::text = end_user OR (auth.role() = 'service_role'));

-- Create policy for inserting request logs
CREATE POLICY "Users can insert their own request logs"
ON public.request_logs
FOR INSERT
WITH CHECK (auth.uid()::text = end_user OR end_user IS NULL OR auth.role() = 'service_role');

-- Create policy for anon insert
CREATE POLICY "Anonymous users can insert request logs"
ON public.request_logs
FOR INSERT
WITH CHECK (end_user IS NULL OR auth.role() = 'anon');
```

These policies are implemented in the `scripts/enable_request_logs_rls.py` script.

## Troubleshooting

### Common Issues

1. **Missing Token Usage Records**
   - Ensure the Supabase client is properly initialized before calling `initialize_litellm_callbacks`
   - Verify that LiteLLM callbacks are properly registered (`litellm.callbacks` should include the Supabase handler)
   - Check if the user ID is being correctly extracted from the session

2. **Duplicate Records**
   - If records appear in both `usage_logs` and `request_logs`, this is expected behavior when `log_to_legacy` is set to `True`
   - To avoid duplicate logging, set `log_to_legacy=False` in the `initialize_app` function once migration is complete

3. **Missing User IDs**
   - Ensure the session mechanism is correctly associating users with requests
   - Check that the `get_user_id_from_session` function is working properly
   - Verify that anonymous user handling is implemented correctly

4. **Permission Errors**
   - Confirm that RLS policies have been correctly applied to the `request_logs` table
   - Ensure users are authenticated properly before accessing their logs
   - Check that service roles have the necessary permissions

### Debugging

Add the following environment variables to enable more detailed logging:

```bash
export LITELLM_LOG_LEVEL=DEBUG  # Enable detailed logging in LiteLLM
export VTAI_DEBUG=1             # Enable debug logging in VT.ai
```

You can also use the Supabase dashboard to directly query the logs and debug any issues:

```sql
-- Check for recent entries in request_logs
SELECT * FROM request_logs ORDER BY created_at DESC LIMIT 10;

-- Check for missing user IDs
SELECT * FROM request_logs WHERE end_user IS NULL ORDER BY created_at DESC LIMIT 10;

-- Compare usage between tables
SELECT COUNT(*) FROM request_logs;
SELECT COUNT(*) FROM usage_logs;
```

### Support

If you encounter persistent issues with token usage monitoring, please:

1. Check the application logs for detailed error messages
2. Review the documentation for any configuration requirements
3. Contact the VT.ai development team with details of the issue

## Summary

The token usage monitoring system in VT.ai provides comprehensive tracking of LLM API usage across different providers. Key features include:

1. **Dual Logging**: Records detailed request information in the new `request_logs` table while maintaining compatibility with the legacy `usage_logs` table.

2. **Optional Legacy Logging**: Can disable logging to the legacy table through the `log_to_legacy` parameter once the migration is complete.

3. **Row-Level Security**: Ensures that users can only access their own usage logs through Supabase RLS policies.

4. **Migration Tools**: Provides both batch and continuous migration tools to transition from the legacy to the new logging system.

5. **Subscription Tier Management**: Integrates with the subscription system to enforce tier-based token limits.

This system enables accurate tracking of token usage per user, which is essential for:

- Managing subscription tiers and billing
- Enforcing usage limits
- Monitoring API costs
- Analyzing usage patterns
- Optimizing resource allocation

By using LiteLLM callbacks, the system captures usage data automatically without requiring changes to the core application logic.
