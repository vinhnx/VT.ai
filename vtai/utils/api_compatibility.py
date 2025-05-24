"""
API compatibility utilities for handling different LLM provider APIs.
"""

import time
from typing import Any, Dict, List, Optional

import litellm
from utils.config import logger
from utils.llm_providers_config import get_llm_params
from utils.supabase_logger import log_request_to_supabase


def is_openai_model(model: str) -> bool:
    """
    Check if a model is an OpenAI model based on naming conventions.

    Args:
        model: The model name to check

    Returns:
        Boolean indicating if the model is likely an OpenAI model
    """
    model_lower = model.lower()
    return (
        any(model_lower.startswith(prefix) for prefix in ["gpt-", "o1", "o3"])
        or "openai" in model_lower
    )


async def try_chat_completion(
    model: str,
    messages: List[Dict[str, Any]],
    user_keys: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Any:
    """
    Use the Chat Completions API directly and log to Supabase.

    Args:
        model: The model to use
        messages: The messages list for the chat completion
        user_keys: Optional user keys for BYOK support
        **kwargs: Additional arguments to pass to the API calls

    Returns:
        The response from the Chat Completions API
    """
    logger.info("Using Chat Completion API for model: %s", model)
    user_id = kwargs.get("user") or "anonymous"
    start_time = time.time()
    try:
        acompletion_kwargs = kwargs.copy()
        if "response_format" not in acompletion_kwargs:
            acompletion_kwargs["response_format"] = {"type": "text"}

        # Get LLM parameters with API keys
        llm_params = get_llm_params(model, user_keys=user_keys)
        acompletion_kwargs.update(llm_params)

        response = await litellm.acompletion(
            model=model, messages=messages, **acompletion_kwargs
        )
        # Log success to Supabase
        log_request_to_supabase(
            model=model,
            messages=messages,
            response=response,
            end_user=user_id,
            status="success",
            response_time=time.time() - start_time,
        )
        return response
    except Exception as e:
        logger.error("Error: %s: %s", type(e).__name__, str(e))
        # Log failure to Supabase
        log_request_to_supabase(
            model=model,
            messages=messages,
            response=None,
            end_user=user_id,
            status="failure",
            error={"error": str(e)},
            response_time=time.time() - start_time,
        )
        raise
