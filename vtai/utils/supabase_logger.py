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


# Token usage is now handled automatically by database triggers
# Monthly usage is calculated on-demand via the monthly_token_usage view
# This function is no longer needed but kept for backward compatibility
def update_user_token_usage(
    user_profile_id: str, 
    tokens_used: int, 
    cost: float, 
    model: str, 
    provider: str
) -> None:
    """
    Update user's token usage tracking.
    
    Note: This is now handled automatically by database triggers.
    The user_profiles.tokens_used field is updated automatically when
    request_logs entries are inserted with a valid user_profile_id.
    """
    logger.debug("Token usage update handled automatically by database trigger for user %s", user_profile_id)


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
        try:
            cost = litellm.completion_cost(completion_response, model)
        except Exception:
            cost = None
        
        # Extract provider from model
        provider = model.split("/")[0] if "/" in model else "openai"
        
        # Log debug info
        logger.info("✅ SUCCESS CALLBACK TRIGGERED: user=%s, model=%s, tokens=%s, call_id=%s", 
                   user, model, tokens_used, litellm_call_id)
        
        log_request_to_supabase(
            model=model,
            messages=messages,
            response=completion_response.model_dump() if hasattr(completion_response, "model_dump") else str(completion_response),
            end_user=user,
            status="success",
            response_time=response_time,
            total_cost=cost,
            litellm_call_id=litellm_call_id,
            user_profile_id=user if user else None,  # Use the session ID as user_profile_id
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
        logger.info("❌ FAILURE CALLBACK TRIGGERED: user=%s, model=%s, error=%s", 
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
        
        # Debug: Check what callbacks are actually set
        logger.info("LiteLLM callbacks configured:")
        logger.info("  Success callbacks: %s", litellm.success_callback)
        logger.info("  Failure callbacks: %s", litellm.failure_callback)
        logger.info("  Supabase URL: %s", SUPABASE_URL[:50] + "...")
    else:
        logger.warning("Cannot setup LiteLLM callbacks: Missing SUPABASE_URL (%s) or SUPABASE_KEY (%s)", 
                      "✓" if SUPABASE_URL else "✗", "✓" if SUPABASE_KEY else "✗")


def get_user_analytics(user_id: str) -> Optional[Dict[str, Any]]:
    """Get comprehensive analytics for a user."""
    if not supabase_client:
        return None
        
    try:
        result = supabase_client.table("user_request_analytics").select("*").eq("user_id", user_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error("Error getting user analytics: %s: %s", type(e).__name__, str(e))
        return None


def get_user_request_history(user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Get user's request history."""
    if not supabase_client:
        return []
        
    try:
        result = supabase_client.rpc("get_user_request_history", {
            "p_user_id": user_id,
            "p_limit": limit,
            "p_offset": offset
        }).execute()
        return result.data or []
    except Exception as e:
        logger.error("Error getting user request history: %s: %s", type(e).__name__, str(e))
        return []


def get_user_token_breakdown(user_id: str) -> List[Dict[str, Any]]:
    """Get user's token usage breakdown by model and provider."""
    if not supabase_client:
        return []
        
    try:
        result = supabase_client.rpc("get_user_token_breakdown", {
            "p_user_id": user_id
        }).execute()
        return result.data or []
    except Exception as e:
        logger.error("Error getting user token breakdown: %s: %s", type(e).__name__, str(e))
        return []


def get_user_monthly_usage(user_id: str) -> List[Dict[str, Any]]:
    """Get user's monthly usage data from the simplified view."""
    if not supabase_client:
        return []
        
    try:
        result = supabase_client.table("monthly_token_usage").select("*").eq("user_profile_id", user_id).execute()
        return result.data or []
    except Exception as e:
        logger.error("Error getting user monthly usage: %s: %s", type(e).__name__, str(e))
        return []


def get_recent_user_activity(limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent activity across all users."""
    if not supabase_client:
        return []
        
    try:
        result = supabase_client.table("user_recent_activity").select("*").limit(limit).execute()
        return result.data or []
    except Exception as e:
        logger.error("Error getting recent user activity: %s: %s", type(e).__name__, str(e))
        return []
