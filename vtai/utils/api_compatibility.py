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
    model: str, messages: List[Dict[str, Any]], **kwargs
) -> Any:
    """
    Use the Chat Completions API directly.

    This replaces the previous try_responses_api_with_fallback function
    that tried Response API first.

    Args:
        model: The model to use
        messages: The messages list for the chat completion
        **kwargs: Additional arguments to pass to the API calls

    Returns:
        The response from the Chat Completions API
    """
    logger.info(f"Using Chat Completion API for model: {model}")

    # Prepare kwargs for acompletion
    acompletion_kwargs = kwargs.copy()

    # Add text response format for consistency
    if "response_format" not in acompletion_kwargs:
        acompletion_kwargs["response_format"] = {"type": "text"}

    # Use messages directly with acompletion
    return await litellm.acompletion(
        model=model, messages=messages, **acompletion_kwargs
    )
