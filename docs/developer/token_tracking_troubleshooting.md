# Token Tracking Troubleshooting Guide

## Common Issues and Solutions

### "APIResponse can't be used in 'await' expression"

**Problem**: You're seeing error messages like "object APIResponse[~_ReturnT] can't be used in 'await' expression" in your logs.

**Cause**: The error occurs when trying to `await` the Supabase Python SDK's `execute()` method. The Supabase Python client's execute() method already returns a response synchronously, not a coroutine, so it can't be awaited.

**Solution**: Remove the `await` keyword when calling `execute()`:

```python
# Incorrect usage - will cause errors:
result = await client.table("request_logs").insert(log_entry).execute()

# Correct usage - execute() is synchronous:
query = client.table("request_logs").insert(log_entry)
result = query.execute()
```

This fix has been applied to the following files:
- `vtai/utils/litellm_callbacks.py`
- `vtai/utils/usage_logger.py`

### Testing Token Tracking

To test if token tracking is working properly:

1. Run the test script with the service key (bypasses RLS policies):
   ```bash
   python scripts/test_fixed_callbacks.py
   ```

2. Check the logs for successful database insertions.

3. Verify in the Supabase dashboard that records are being created in both `request_logs` and `usage_logs` tables.

### Database Access Issues

**Problem**: Insertions fail with 401 Unauthorized errors.

**Cause**: Row-Level Security (RLS) policies may be preventing the service from inserting records.

**Solution**:
1. Use the Supabase service key instead of the regular key:
   ```python
   supabase_client = create_client(supabase_url, os.environ.get("SUPABASE_SERVICE_KEY"))
   ```

2. Or update the RLS policies to allow the application to insert records:
   ```sql
   CREATE POLICY "Allow service role to insert request_logs"
   ON public.request_logs
   FOR INSERT
   TO service_role
   WITH CHECK (true);
   ```

## Callback Registration Issues

**Problem**: Token usage is not being logged despite no errors.

**Cause**: LiteLLM callbacks may not be properly registered.

**Solution**:
1. Ensure the initialization function is called:
   ```python
   initialize_litellm_callbacks(supabase_client, log_to_legacy=True)
   ```

2. Verify that callbacks are registered:
   ```python
   print(f"Callbacks registered: {litellm.callbacks}")
   ```

3. The VTAISupabaseHandler should implement all necessary callback methods, including `__call__`, `log_success_event`, and `log_failure_event`.

## Manual Testing

You can run these diagnostic scripts to troubleshoot:

1. Test basic connectivity to Supabase:
   ```bash
   python scripts/test_direct_insertion.py
   ```

2. Test the full callback mechanism:
   ```bash
   python scripts/test_fixed_callbacks.py
   ```

3. Debug the callback registration:
   ```bash
   python scripts/debug_litellm_callbacks.py
   ```
