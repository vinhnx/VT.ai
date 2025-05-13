# Token Tracking: Usage Logs Removal

## Overview

As of May 13, 2025, the token tracking system has been simplified by removing the legacy `usage_logs` table. All token usage data is now stored exclusively in the `request_logs` table.

## Changes Made

1. **Removed `usage_logs` Table**:
   - The `usage_logs` table has been completely removed from the database
   - All token tracking now exclusively uses the more comprehensive `request_logs` table

2. **Removed Legacy Integration**:
   - Removed the `log_to_legacy` parameter from all functions
   - Removed the `_log_to_legacy_table` method from the `VTAISupabaseHandler` class
   - Removed all references to the legacy table in the documentation

3. **Updated Function Signatures**:
   - `initialize_litellm_callbacks(supabase_client)` no longer accepts the `log_to_legacy` parameter
   - All calls to this function have been updated to remove the parameter

4. **Simplified RLS Policies**:
   - Cleaned up redundant RLS policies for the `request_logs` table
   - Created a clear set of policies:
     - Authenticated users can insert records
     - Anonymous users can insert records
     - Users can only view their own records
     - Service role has full access to all records

## Developer Impact

### Code Changes

If you have code that uses the `initialize_litellm_callbacks` function, you need to update it:

```python
# Old (no longer supported)
initialize_litellm_callbacks(supabase_client, log_to_legacy=True)

# New
initialize_litellm_callbacks(supabase_client)
```

### Data Access

If your application was querying the `usage_logs` table directly, you need to update your queries to use the `request_logs` table instead:

```sql
-- Old (no longer works)
SELECT * FROM usage_logs WHERE user_id = 'user-123';

-- New
SELECT * FROM request_logs WHERE end_user = 'user-123';
```

### Field Mapping

When migrating queries from `usage_logs` to `request_logs`, use the following field mapping:

| usage_logs Field | request_logs Field or Path |
|------------------|----------------------------|
| user_id          | end_user                   |
| session_id       | additional_details->>'session_id' |
| model_name       | model                      |
| input_tokens     | additional_details->>'input_tokens' |
| output_tokens    | additional_details->>'output_tokens' |
| total_tokens     | additional_details->>'total_tokens' |
| cost             | total_cost                 |
| created_at       | created_at                 |
| auth_method      | additional_details->>'auth_method' |

## Benefits

1. **Simplified Architecture**: One source of truth for token usage data
2. **Improved Performance**: Fewer database writes and no redundant storage
3. **Better Data Integrity**: No risk of inconsistencies between tables
4. **Streamlined Code**: Cleaner, more maintainable codebase
5. **Clearer Permissions Model**: Simplified RLS policies are easier to understand and maintain
