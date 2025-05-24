"""
Supabase logging integration for LiteLLM callbacks.

Logs LLM requests and responses to Supabase for both success and failure events.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import litellm
from supabase import Client as SupabaseClient
from supabase import create_client

from vtai.utils.config import logger

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

# We'll use custom callback functions instead of LiteLLM's built-in Supabase integration

supabase_client: Optional[SupabaseClient] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    logger.warning("Supabase credentials not set. Logging will be disabled.")


def log_request_to_supabase(
    model: str,
    messages: Any,
    response: Any,
    end_user: str,
    status: str,
    error: Any = None,
    response_time: Optional[float] = None,
    total_cost: Optional[float] = None,
    additional_details: Optional[Dict[str, Any]] = None,
    litellm_call_id: Optional[str] = None,
    user_profile_id: Optional[str] = None,
    tokens_used: Optional[int] = None,
    provider: Optional[str] = None,
) -> None:
    """Log a request to the Supabase request_logs table."""
    if not supabase_client:
        logger.warning("Supabase client not initialized. Skipping log.")
        return
    try:
        # Check if user profile exists if user_profile_id is provided
        if user_profile_id:
            try:
                user_check = supabase_client.table("user_profiles").select("user_id").eq("user_id", user_profile_id).execute()
                if not user_check.data:
                    logger.warning("User profile %s not found, logging without user link", user_profile_id)
                    user_profile_id = None
            except Exception:
                logger.warning("Could not verify user profile %s, logging without user link", user_profile_id)
                user_profile_id = None
        
        row = {
            "model": model or "",
            "messages": messages or {},
            "response": response or {},
            "end_user": end_user or "",
            "status": status or "",
            "error": error or {},
            "response_time": response_time or 0.0,
            "total_cost": total_cost,
            "additional_details": additional_details or {},
            "litellm_call_id": litellm_call_id,
            "user_profile_id": user_profile_id,
            "tokens_used": tokens_used or 0,
            "provider": provider or "",
        }
        supabase_client.table("request_logs").insert(row).execute()
        
        # Update user token usage if user_profile_id is provided
        if user_profile_id and tokens_used:
            try:
                update_user_token_usage(user_profile_id, tokens_used, total_cost or 0.0, model, provider)
            except Exception as token_error:
                logger.warning("Failed to update token usage for user %s: %s", user_profile_id, str(token_error))
            
    except Exception as e:
        logger.error("Error logging to Supabase: %s: %s", type(e).__name__, str(e))


def update_user_token_usage(
    user_profile_id: str, 
    tokens_used: int, 
    cost: float, 
    model: str, 
    provider: str
) -> None:
    """Update user's token usage tracking."""
    if not supabase_client:
        return
        
    try:
        # Update user_profiles tokens_used counter
        current_result = supabase_client.table("user_profiles").select("tokens_used").eq("user_id", user_profile_id).execute()
        current_tokens = current_result.data[0]["tokens_used"] if current_result.data else 0
        new_total = current_tokens + tokens_used
        
        supabase_client.table("user_profiles").update({"tokens_used": new_total}).eq("user_id", user_profile_id).execute()
        
        # Update monthly usage tracking
        now = datetime.now()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Try to get existing monthly record
        monthly_result = supabase_client.table("tokens_usage").select("*").eq("user_profile_id", user_profile_id).eq("period_type", "monthly").eq("period_start", period_start.isoformat()).execute()
        
        if monthly_result.data:
            # Update existing record
            existing = monthly_result.data[0]
            new_tokens = existing["total_tokens"] + tokens_used
            new_cost = existing["total_cost"] + cost
            
            # Update breakdowns
            model_breakdown = existing.get("model_breakdown", {})
            model_breakdown[model] = model_breakdown.get(model, 0) + tokens_used
            
            provider_breakdown = existing.get("provider_breakdown", {})
            provider_breakdown[provider] = provider_breakdown.get(provider, 0) + tokens_used
            
            supabase_client.table("tokens_usage").update({
                "total_tokens": new_tokens,
                "total_cost": new_cost,
                "model_breakdown": model_breakdown,
                "provider_breakdown": provider_breakdown,
                "updated_at": now.isoformat()
            }).eq("id", existing["id"]).execute()
        else:
            # Create new monthly record
            supabase_client.table("tokens_usage").insert({
                "user_profile_id": user_profile_id,
                "total_tokens": tokens_used,
                "total_cost": cost,
                "period_start": period_start.isoformat(),
                "period_type": "monthly",
                "model_breakdown": {model: tokens_used},
                "provider_breakdown": {provider: tokens_used}
            }).execute()
            
    except Exception as e:
        logger.error("Error updating token usage: %s: %s", type(e).__name__, str(e))


# LiteLLM callback functions
def success_callback_supabase(
    kwargs: Dict[str, Any],
    completion_response: Any,
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Success callback for LiteLLM to log successful requests."""
    try:
        # Extract relevant information
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        user = kwargs.get("user", "")
        litellm_call_id = kwargs.get("litellm_call_id")
        
        # Calculate response time
        response_time = (end_time - start_time).total_seconds() if start_time and end_time else 0
        
        # Extract tokens and cost information
        usage = getattr(completion_response, "usage", None)
        tokens_used = usage.total_tokens if usage else 0
        
        # Get cost information
        cost = litellm.completion_cost(completion_response, model)
        
        # Extract provider from model
        provider = model.split("/")[0] if "/" in model else "openai"
        
        log_request_to_supabase(
            model=model,
            messages=messages,
            response=completion_response.model_dump() if hasattr(completion_response, "model_dump") else str(completion_response),
            end_user=user,
            status="success",
            response_time=response_time,
            total_cost=cost,
            litellm_call_id=litellm_call_id,
            user_profile_id=user,
            tokens_used=tokens_used,
            provider=provider
        )
        
    except Exception as e:
        logger.error("Error in success callback: %s: %s", type(e).__name__, str(e))


def failure_callback_supabase(
    kwargs: Dict[str, Any],
    completion_response: Any,
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Failure callback for LiteLLM to log failed requests."""
    try:
        # Extract relevant information
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        user = kwargs.get("user", "")
        litellm_call_id = kwargs.get("litellm_call_id")
        
        # Calculate response time
        response_time = (end_time - start_time).total_seconds() if start_time and end_time else 0
        
        # Extract provider from model
        provider = model.split("/")[0] if "/" in model else "openai"
        
        # Log the error details
        error_details = {"error": str(completion_response)} if completion_response else {}
        logger.info("Logging failure callback for user: %s, model: %s, error: %s", 
                   user, model, str(completion_response)[:100])
        
        log_request_to_supabase(
            model=model,
            messages=messages,
            response={},
            end_user=user,
            status="error",
            error=error_details,
            response_time=response_time,
            litellm_call_id=litellm_call_id,
            user_profile_id=user if user else None,
            provider=provider
        )
        
    except Exception as e:
        logger.error("Error in failure callback: %s: %s", type(e).__name__, str(e))


def setup_litellm_callbacks():
    """Setup LiteLLM callbacks for Supabase logging."""
    if SUPABASE_URL and SUPABASE_KEY:
        # Use our custom callback functions instead of LiteLLM's built-in ones
        # This avoids the credential issues with LiteLLM's Supabase integration
        litellm.success_callback = [success_callback_supabase]
        litellm.failure_callback = [failure_callback_supabase]
        
        logger.info("LiteLLM custom Supabase callbacks configured successfully with URL: %s", SUPABASE_URL[:50] + "...")
    else:
        logger.warning("Cannot setup LiteLLM callbacks: Missing SUPABASE_URL (%s) or SUPABASE_KEY (%s)", 
                      "✓" if SUPABASE_URL else "✗", "✓" if SUPABASE_KEY else "✗")
