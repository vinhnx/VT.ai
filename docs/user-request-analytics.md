# User-Request Analytics System

This document describes the comprehensive user analytics system that connects request logs to user profiles, enabling detailed tracking and analysis of LLM usage patterns.

## Database Relationships

### Core Tables

#### `user_profiles` (Primary)
- **Primary Key**: `user_id` (text)
- **Purpose**: Store user account information and aggregated stats
- **Key Fields**: `email`, `full_name`, `tokens_used`, `subscription_tier`

#### `request_logs` (Linked)
- **Primary Key**: `id` (bigint)
- **Foreign Key**: `user_profile_id` → `user_profiles.user_id`
- **Purpose**: Detailed log of every LLM request
- **Key Fields**: `model`, `status`, `tokens_used`, `total_cost`, `provider`

#### `tokens_usage` (Aggregated)
- **Primary Key**: `id` (bigint)
- **Foreign Key**: `user_profile_id` → `user_profiles.user_id`
- **Purpose**: Monthly usage aggregation and breakdown
- **Key Fields**: `total_tokens`, `total_cost`, `model_breakdown`, `provider_breakdown`

### Relationship Structure
```
user_profiles (1) ←→ (N) request_logs
user_profiles (1) ←→ (N) tokens_usage
```

## Analytics Views

### 1. `user_request_analytics` View
Comprehensive user statistics combining profile and request data.

```sql
SELECT * FROM user_request_analytics WHERE user_id = 'user_123';
```

**Returns:**
- `total_requests` - Total number of LLM requests
- `successful_requests` - Number of successful requests  
- `failed_requests` - Number of failed requests
- `total_tokens_from_logs` - Total tokens consumed
- `total_cost` - Total cost across all requests
- `avg_response_time` - Average response time
- `most_used_model` - Most frequently used model
- `most_used_provider` - Most frequently used provider
- `last_request_time` - Timestamp of last request

### 2. `user_recent_activity` View
Recent activity across all users with user details.

```sql
SELECT * FROM user_recent_activity ORDER BY request_time DESC LIMIT 20;
```

**Returns:** Combined user profile and request log data for recent activity monitoring.

## Analytics Functions

### 1. `get_user_request_history(user_id, limit, offset)`
Paginated request history for a specific user.

```sql
SELECT * FROM get_user_request_history('user_123', 50, 0);
```

**Parameters:**
- `user_id` (text) - User identifier
- `limit` (int) - Maximum records to return (default: 50)
- `offset` (int) - Records to skip for pagination (default: 0)

### 2. `get_user_token_breakdown(user_id)`
Token usage breakdown by model and provider.

```sql
SELECT * FROM get_user_token_breakdown('user_123');
```

**Returns:**
- `model` - LLM model name
- `provider` - Provider (openai, anthropic, etc.)
- `request_count` - Number of requests for this model
- `total_tokens` - Total tokens consumed
- `total_cost` - Total cost for this model
- `avg_tokens_per_request` - Average tokens per request
- `last_used` - Last time this model was used

## Python API

### Analytics Functions

```python
from vtai.utils.supabase_logger import (
    get_user_analytics,
    get_user_request_history, 
    get_user_token_breakdown,
    get_recent_user_activity
)

# Get comprehensive user analytics
analytics = get_user_analytics('user_123')
print(f"Total requests: {analytics['total_requests']}")
print(f"Total cost: ${analytics['total_cost']}")

# Get request history with pagination
history = get_user_request_history('user_123', limit=20, offset=0)
for request in history:
    print(f"{request['model']}: {request['status']} - {request['tokens_used']} tokens")

# Get token breakdown by model/provider
breakdown = get_user_token_breakdown('user_123')
for item in breakdown:
    print(f"{item['model']} ({item['provider']}): {item['total_tokens']} tokens")

# Get recent activity across all users
activity = get_recent_user_activity(limit=10)
for item in activity:
    print(f"{item['email']}: {item['model']} - {item['status']}")
```

## Usage Examples

### User Dashboard Data
```python
def get_user_dashboard_data(user_id: str):
    """Get all data needed for a user dashboard."""
    analytics = get_user_analytics(user_id)
    recent_requests = get_user_request_history(user_id, limit=10)
    token_breakdown = get_user_token_breakdown(user_id)
    
    return {
        'summary': analytics,
        'recent_activity': recent_requests,
        'usage_breakdown': token_breakdown
    }
```

### Admin Analytics
```python
def get_admin_overview():
    """Get overview data for admin dashboard."""
    recent_activity = get_recent_user_activity(limit=50)
    
    # Get all user analytics
    result = supabase_client.table("user_request_analytics").select("*").execute()
    all_users = result.data
    
    return {
        'recent_activity': recent_activity,
        'user_summaries': all_users,
        'total_users': len(all_users),
        'active_users': len([u for u in all_users if u['total_requests'] > 0])
    }
```

### Cost Analysis
```python
def analyze_user_costs(user_id: str):
    """Analyze cost patterns for a user."""
    breakdown = get_user_token_breakdown(user_id)
    
    total_cost = sum(float(item['total_cost'] or 0) for item in breakdown)
    most_expensive = max(breakdown, key=lambda x: float(x['total_cost'] or 0))
    
    return {
        'total_cost': total_cost,
        'most_expensive_model': most_expensive['model'],
        'cost_by_provider': {},  # Could aggregate by provider
        'recommendations': generate_cost_recommendations(breakdown)
    }
```

## Security & RLS

### Row Level Security Policies

All tables have RLS enabled with these policies:

**`request_logs`:**
- Users can only see their own request logs
- Service role has full access for system operations
- Anonymous users can insert logs (for callback functionality)

**`tokens_usage`:**  
- Users can only see their own usage data
- Service role has full access
- Users can insert/update their own usage records

**`user_profiles`:**
- Users can only see/modify their own profile
- Service role has full access

### Access Control
```python
# User can only access their own data
analytics = get_user_analytics(current_user.id)  # ✅ Allowed

# Admin/service role can access any user's data  
analytics = get_user_analytics(any_user_id)  # ✅ Allowed if service role
```

## Performance Considerations

### Indexing
- `request_logs.user_profile_id` - Indexed for fast user lookups
- `request_logs.created_at` - Indexed for time-based queries
- `tokens_usage.user_profile_id` - Indexed for user aggregations

### Query Optimization
- Views use optimized subqueries for model/provider lookups
- Functions use proper LIMIT/OFFSET for pagination
- Aggregations are computed in views to avoid repeated calculations

### Caching Recommendations
- Cache user analytics for 5-10 minutes
- Cache token breakdowns for 1 hour
- Real-time data only for recent activity

## Monitoring & Alerts

### Usage Thresholds
```python
def check_user_usage_limits(user_id: str):
    """Check if user is approaching usage limits."""
    analytics = get_user_analytics(user_id)
    
    if analytics['total_cost'] > 10.0:  # $10 threshold
        send_usage_alert(user_id, 'cost', analytics['total_cost'])
    
    if analytics['total_tokens_from_logs'] > 100000:  # 100k tokens
        send_usage_alert(user_id, 'tokens', analytics['total_tokens_from_logs'])
```

### Health Monitoring
```python
def monitor_system_health():
    """Monitor overall system health."""
    recent_activity = get_recent_user_activity(limit=100)
    
    error_rate = len([a for a in recent_activity if a['status'] in ['error', 'failure']]) / len(recent_activity)
    
    if error_rate > 0.1:  # 10% error rate
        alert_admin('High error rate detected', error_rate)
```

---

This analytics system provides comprehensive insights into user behavior, usage patterns, and system performance while maintaining proper security and data isolation through RLS policies.