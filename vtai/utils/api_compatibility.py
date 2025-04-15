"""
API compatibility utilities for handling different LLM provider APIs.
"""
import asyncio
import functools
import inspect
from typing import Any, Dict, List, Optional

import litellm
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    ServiceUnavailableError,
)

from vtai.utils.config import logger

# Cache for models that support Responses API to avoid repeated checks
_RESPONSES_API_SUPPORT_CACHE = {}

# Known OpenAI models that support Responses API
KNOWN_OPENAI_RESPONSE_MODELS = {
    "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o",
    "o1", "o1-mini", "o1-preview", "o3"
}

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
        any(model_lower.startswith(prefix) for prefix in ["gpt-", "o1", "o3"]) or
        "openai" in model_lower
    )

def reset_responses_api_cache():
    """Clear the cached results of model compatibility checks."""
    _RESPONSES_API_SUPPORT_CACHE.clear()
    logger.debug("Responses API support cache has been cleared")

async def check_responses_api_support(model: str) -> bool:
    """
    Asynchronously check if a model supports the Responses API by making a test call.

    Args:
        model: The model name to check

    Returns:
        Boolean indicating if the model supports the Responses API
    """
    # Return from cache if available
    if model in _RESPONSES_API_SUPPORT_CACHE:
        return _RESPONSES_API_SUPPORT_CACHE[model]

    # Fast path for known OpenAI models that support Responses API
    if any(known_model in model.lower() for known_model in KNOWN_OPENAI_RESPONSE_MODELS):
        _RESPONSES_API_SUPPORT_CACHE[model] = True
        return True

    # For non-OpenAI models, we need to attempt a test call
    if not is_openai_model(model):
        # Most non-OpenAI models currently don't support the Responses API
        _RESPONSES_API_SUPPORT_CACHE[model] = False
        return False

    # For unknown OpenAI models, make a test call
    try:
        # Use a minimal test input to check API compatibility
        test_input = [{"role": "user", "content": "test"}]

        # Set a short timeout for the test request
        await asyncio.wait_for(
            litellm.responses(
                model=model,
                input=test_input,
                max_output_tokens=1,  # Minimize token usage
                timeout=5.0,  # Short timeout for the check
            ),
            timeout=8.0,  # Overall timeout including network overhead
        )

        # If we get here, the model supports Responses API
        _RESPONSES_API_SUPPORT_CACHE[model] = True
        logger.info(f"Model {model} supports Responses API")
        return True

    except (ServiceUnavailableError, AuthenticationError) as e:
        # Authentication or service errors should not be considered API compatibility issues
        # Don't cache these results as they might be temporary
        logger.warning(f"Service error while checking Responses API support for {model}: {e}")
        raise

    except Exception as e:
        # Any other exception indicates the model doesn't support Responses API
        _RESPONSES_API_SUPPORT_CACHE[model] = False
        logger.debug(f"Model {model} does not support Responses API: {e}")
        return False

def supports_responses_api(model: str) -> bool:
    """
    Determines if the model supports the Responses API.

    This is a synchronous wrapper around the async check function.
    For most use cases, prefer using try_responses_api_with_fallback instead.

    Args:
        model: The model name to check

    Returns:
        Boolean indicating if the model supports the Responses API
    """
    # Fast path for cached results
    if model in _RESPONSES_API_SUPPORT_CACHE:
        return _RESPONSES_API_SUPPORT_CACHE[model]

    # Fast path for known OpenAI models
    if any(known_model in model.lower() for known_model in KNOWN_OPENAI_RESPONSE_MODELS):
        _RESPONSES_API_SUPPORT_CACHE[model] = True
        return True

    # For unknown models, just check if it's an OpenAI model
    # This is less accurate but avoids async issues in sync contexts
    supports = is_openai_model(model)
    _RESPONSES_API_SUPPORT_CACHE[model] = supports
    return supports

async def try_responses_api_with_fallback(
    model: str,
    messages: List[Dict[str, Any]],
    **kwargs
) -> Any:
    """
    Try to use the Responses API first, falling back to Chat Completions API if needed.

    Args:
        model: The model to use
        messages: The messages list for the chat completion
        **kwargs: Additional arguments to pass to the API calls

    Returns:
        The response from either the Responses API or Chat Completions API
    """
    # Always try Responses API first for potentially supported models
    try:
        logger.info(f"Trying Responses API for model: {model}")
        # For Responses API, 'messages' should be passed as 'input'
        kwargs_copy = kwargs.copy()

        # Convert 'messages' to 'input' for Responses API
        if "input" not in kwargs_copy:
            kwargs_copy["input"] = messages

        # Remove parameters not supported by Responses API
        if "response_format" in kwargs_copy:
            del kwargs_copy["response_format"]

        return await litellm.responses(model=model, **kwargs_copy)

    except Exception as e:
        # Log the error and fall back to Chat Completions API
        logger.info(f"Falling back to Chat Completions API for model {model}. Error: {e}")

        # Prepare kwargs for acompletion
        acompletion_kwargs = kwargs.copy()

        # Add text response format for consistency with Responses API
        if "response_format" not in acompletion_kwargs:
            acompletion_kwargs["response_format"] = {"type": "text"}

        # For acompletion, use messages directly
        return await litellm.acompletion(model=model, messages=messages, **acompletion_kwargs)            acompletion_kwargs["response_format"] = {"type": "text"}

        # For acompletion, use messages directly
        return await litellm.acompletion(model=model, messages=messages, **acompletion_kwargs)