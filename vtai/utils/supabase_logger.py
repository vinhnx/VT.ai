"""
Supabase logging integration for LiteLLM callbacks.

Logs LLM requests and responses to Supabase for both success and failure events.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import litellm
from supabase import Client as SupabaseClient
from supabase import create_client
from utils.config import logger

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

# We'll use custom callback functions instead of LiteLLM's built-in Supabase integration

supabase_client: Optional[SupabaseClient] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    logger.warning("Supabase credentials not set. Logging will be disabled.")


def calculate_token_costs(
    model: str, prompt_tokens: int, completion_tokens: int, _total_tokens: int
) -> Tuple[float, float, float]:
    """Calculate detailed token costs using LiteLLM's cost functions."""
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        total_cost = (prompt_tokens * prompt_cost) + (
            completion_tokens * completion_cost
        )
        return prompt_cost, completion_cost, total_cost
    except (ValueError, TypeError) as e:
        logger.error("Error: %s: %s", type(e).__name__, str(e))
        return 0.0, 0.0, 0.0


def get_model_info(model: str) -> Dict[str, Any]:
    """Get detailed model information including max tokens and cost structure."""
    try:
        model_cost_dict = litellm.model_cost
        model_info = model_cost_dict.get(model, {})
        try:
            max_tokens = litellm.get_max_tokens(model)
        except (ValueError, TypeError) as e:
            logger.error("Error: %s: %s", type(e).__name__, str(e))
            max_tokens = model_info.get("max_tokens", 4096)
        return {
            "model": model,
            "max_tokens": max_tokens,
            "input_cost_per_token": model_info.get("input_cost_per_token"),
            "output_cost_per_token": model_info.get("output_cost_per_token"),
            "litellm_provider": model_info.get("litellm_provider"),
            "mode": model_info.get("mode", "chat"),
        }
    except (ValueError, TypeError) as e:
        logger.error("Error: %s: %s", type(e).__name__, str(e))
        return {"model": model, "max_tokens": 4096}


def calculate_prompt_tokens(model: str, messages: List[Dict[str, Any]]) -> int:
    """Calculate token count for prompt messages using LiteLLM's token_counter."""
    try:
        return litellm.token_counter(model=model, messages=messages)
    except (ValueError, TypeError) as e:
        logger.error("Error: %s: %s", type(e).__name__, str(e))
        return 0


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
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    prompt_cost_per_token: Optional[float] = None,
    completion_cost_per_token: Optional[float] = None,
) -> None:
    """Log a request to the Supabase request_logs table."""
    if not supabase_client:
        logger.warning("Supabase client not initialized. Skipping log.")
        return
    try:
        if user_profile_id:
            try:
                user_check = (
                    supabase_client.table("user_profiles")
                    .select("user_id")
                    .eq("user_id", user_profile_id)
                    .execute()
                )
                if not user_check.data:
                    logger.warning(
                        "User profile %s not found, logging without user link",
                        user_profile_id,
                    )
                    user_profile_id = None
            # ruff: noqa: E722 - bare except required for external API/DB robustness
            except Exception as e:
                logger.error("Error: %s: %s", type(e).__name__, str(e))
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
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "prompt_cost_per_token": prompt_cost_per_token,
            "completion_cost_per_token": completion_cost_per_token,
        }
        supabase_client.table("request_logs").insert(row).execute()
        # No update_user_token_usage call (handled by DB trigger)
    # ruff: noqa: E722 - bare except required for external API/DB robustness
    except Exception as e:
        logger.error("Error: %s: %s", type(e).__name__, str(e))


def update_user_token_usage(
    user_profile_id: str, _tokens_used: int, _cost: float, _model: str, _provider: str
) -> None:
    """
    Update user's token usage tracking.
    Note: This is now handled automatically by database triggers.
    """
    logger.debug(
        "Token usage update handled automatically by database trigger for user %s",
        user_profile_id,
    )


def fetch_user_profile_from_supabase(user_id: str) -> dict:
    """Fetch user profile from Supabase database by user_id."""
    if not supabase_client:
        return {}
    try:
        response = (
            supabase_client.table("user_profiles")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )
        if response.data and len(response.data) > 0:
            return response.data[0]
        return {}
    # ruff: noqa: E722 - bare except required for external API/DB robustness
    except Exception as e:
        logger.error("Error fetching user profile: %s: %s", type(e).__name__, str(e))
        return {}


# LiteLLM callback functions
def success_callback_supabase(
    kwargs: Dict[str, Any],
    completion_response: Any,
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Success callback for LiteLLM to log successful requests with detailed cost breakdown."""
    try:
        # Only log if HTTP status is 200 (success)
        status_code = getattr(completion_response, "status_code", 200)
        if status_code != 200:
            logger.info("Skipping Supabase log: non-200 status (%s)", status_code)
            return

        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        user = kwargs.get("user", "")
        litellm_call_id = kwargs.get("litellm_call_id")
        response_time = (
            (end_time - start_time).total_seconds() if start_time and end_time else 0
        )

        # Extract token usage, fallback to token_counter if missing
        usage = getattr(completion_response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        fallback_used = False
        if prompt_tokens is None or completion_tokens is None or total_tokens is None:
            # Fallback: estimate tokens using LiteLLM's token_counter
            prompt_tokens = calculate_prompt_tokens(model, messages)
            completion_tokens = 0
            total_tokens = prompt_tokens
            fallback_used = True

        # Calculate cost using LiteLLM helpers
        prompt_cost_per_token, completion_cost_per_token, calculated_cost = (
            calculate_token_costs(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                _total_tokens=total_tokens,
            )
        )
        # Also get LiteLLM's built-in cost calculation as fallback
        try:
            litellm_cost = litellm.completion_cost(
                completion_response=completion_response
            )
        except Exception:
            litellm_cost = None
        final_cost = calculated_cost if calculated_cost > 0 else (litellm_cost or 0.0)
        if calculated_cost == 0 and litellm_cost is not None:
            fallback_used = True

        provider = model.split("/")[0] if "/" in model else "openai"
        # Gather model metadata (from LiteLLM's model_cost)
        model_info = get_model_info(model)

        # Build detailed breakdown for analytics/debugging
        cost_breakdown = {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total_tokens or 0,
            "prompt_cost_per_token": prompt_cost_per_token,
            "completion_cost_per_token": completion_cost_per_token,
            "calculated_cost": calculated_cost,
            "litellm_cost": litellm_cost,
            "final_cost": final_cost,
            "fallback_used": fallback_used,
            "model_info": model_info,
        }

        logger.info(
            "✅ SUCCESS CALLBACK: user=%s, model=%s, prompt_tokens=%s, completion_tokens=%s, total_tokens=%s, cost=$%.6f, call_id=%s, fallback=%s",
            user,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            final_cost or 0,
            litellm_call_id,
            fallback_used,
        )

        log_request_to_supabase(
            model=model,
            messages=messages,
            response=(
                completion_response.model_dump()
                if hasattr(completion_response, "model_dump")
                else str(completion_response)
            ),
            end_user=user,
            status="success",
            response_time=response_time,
            total_cost=final_cost,
            litellm_call_id=litellm_call_id,
            user_profile_id=user if user else None,
            tokens_used=total_tokens or 0,
            provider=provider,
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            prompt_cost_per_token=prompt_cost_per_token,
            completion_cost_per_token=completion_cost_per_token,
            additional_details=cost_breakdown,
        )

    # ruff: noqa: E722 - bare except required for external API/DB robustness
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
        response_time = (
            (end_time - start_time).total_seconds() if start_time and end_time else 0
        )

        # Extract provider from model
        provider = model.split("/")[0] if "/" in model else "openai"

        # Log the error details
        error_details = (
            {"error": str(completion_response)} if completion_response else {}
        )
        logger.info(
            "❌ FAILURE CALLBACK TRIGGERED: user=%s, model=%s, error=%s",
            user,
            model,
            str(completion_response)[:100],
        )

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
            provider=provider,
        )

    # ruff: noqa: E722 - bare except required for external API/DB robustness
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
        logger.warning(
            "Cannot setup LiteLLM callbacks: Missing SUPABASE_URL (%s) or SUPABASE_KEY (%s)",
            "✓" if SUPABASE_URL else "✗",
            "✓" if SUPABASE_KEY else "✗",
        )


def get_user_analytics(user_id: str) -> Optional[Dict[str, Any]]:
    """Get comprehensive analytics for a user."""
    if not supabase_client:
        return None

    try:
        result = (
            supabase_client.table("user_request_analytics")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )
        return result.data[0] if result.data else None
    # ruff: noqa: E722 - bare except required for external API/DB robustness
    except Exception as e:
        logger.error("Error getting user analytics: %s: %s", type(e).__name__, str(e))
        return None


def get_user_request_history(
    user_id: str, limit: int = 50, offset: int = 0
) -> List[Dict[str, Any]]:
    """Get user's request history."""
    if not supabase_client:
        return []

    try:
        result = supabase_client.rpc(
            "get_user_request_history",
            {"p_user_id": user_id, "p_limit": limit, "p_offset": offset},
        ).execute()
        return result.data or []
    # ruff: noqa: E722 - bare except required for external API/DB robustness
    except Exception as e:
        logger.error(
            "Error getting user request history: %s: %s", type(e).__name__, str(e)
        )
        return []


def get_user_token_breakdown(user_id: str) -> List[Dict[str, Any]]:
    """Get user's token usage breakdown by model and provider."""
    if not supabase_client:
        return []

    try:
        result = supabase_client.rpc(
            "get_user_token_breakdown", {"p_user_id": user_id}
        ).execute()
        return result.data or []
    # ruff: noqa: E722 - bare except required for external API/DB robustness
    except Exception as e:
        logger.error(
            "Error getting user token breakdown: %s: %s", type(e).__name__, str(e)
        )
        return []


def get_user_monthly_usage(user_id: str) -> List[Dict[str, Any]]:
    """Get user's monthly usage data from the simplified view."""
    if not supabase_client:
        return []

    try:
        result = (
            supabase_client.table("monthly_token_usage")
            .select("*")
            .eq("user_profile_id", user_id)
            .execute()
        )
        return result.data or []
    # ruff: noqa: E722 - bare except required for external API/DB robustness
    except Exception as e:
        logger.error(
            "Error getting user monthly usage: %s: %s", type(e).__name__, str(e)
        )
        return []


def get_recent_user_activity(limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent activity across all users."""
    if not supabase_client:
        return []

    try:
        result = (
            supabase_client.table("user_recent_activity")
            .select("*")
            .limit(limit)
            .execute()
        )
        return result.data or []
    # ruff: noqa: E722 - bare except required for external API/DB robustness
    except Exception as e:
        logger.error(
            "Error getting recent user activity: %s: %s", type(e).__name__, str(e)
        )
        return []
