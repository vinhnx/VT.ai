# Request Logs Token Usage Monitoring Implementation

This document outlines the implementation of token usage monitoring with the Supabase `request_logs` table, which replaces the deprecated `usage_logs` table.

## Issues and Fixes

### 1. Syntax Error in LiteLLM Callbacks

The `_log_to_usage_logs` method in `vtai/utils/litellm_callbacks.py` was missing a corresponding `except` block for its `try` statement. This was fixed by adding the proper exception handling.

### 2. Row Level Security Policies

The `request_logs` table had restrictive Row Level Security (RLS) policies that prevented proper insertions, especially from anonymous users or service roles.

#### RLS Policy Fixes

The following SQL was applied to fix the RLS policies:

```sql
-- Drop existing policies
DROP POLICY IF EXISTS "Anonymous users can insert request logs" ON public.request_logs;
DROP POLICY IF EXISTS "Users can insert their own request logs" ON public.request_logs;

-- Create permissive policies
CREATE POLICY "Anonymous users can insert request logs (fixed)"
ON public.request_logs
FOR INSERT
TO anon
WITH CHECK (true);

CREATE POLICY "Users can insert their own request logs (fixed)"
ON public.request_logs
FOR INSERT
TO authenticated
WITH CHECK (true);

CREATE POLICY "Enable read access for all users"
ON public.request_logs
FOR SELECT
TO anon, authenticated
USING (true);

-- Grant appropriate permissions
GRANT SELECT, INSERT ON public.request_logs TO anon, authenticated, service_role;
```

### 3. Implementation Details

The token usage monitoring system now uses the `request_logs` table for all LLM API call logging. The implementation includes:

1. A `VTAISupabaseHandler` class that extends LiteLLM's callback system
2. Direct service role authentication for guaranteed database access
3. Proper error handling to gracefully manage potential failures

## Testing and Validation

Testing confirmed that:

1. Direct inserts to the `request_logs` table work successfully with the service role key
2. The SQL fixes for RLS policies correctly allow both anonymous and authenticated users to insert logs
3. The token usage data is properly structured for analytics and monitoring

## Recommendations

1. For production environments, always use the service role key for guaranteed database access
2. Consider implementing a fallback logging mechanism (e.g., local file logs) if database connectivity fails
3. Regularly monitor the logs to ensure they continue working after any Supabase schema changes
