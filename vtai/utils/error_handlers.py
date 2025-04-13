"""
Error handling utilities for VT application.

Provides standardized error handling and user feedback for various exception types.
"""

import chainlit as cl
from litellm.exceptions import BadRequestError, RateLimitError, ServiceUnavailableError

from vtai.utils.config import logger


async def handle_exception(e: Exception) -> None:
    """
    Handles exceptions that occur during LLM interactions with specific error messages
    based on the type of exception.

    Args:
        e: The exception that was raised
    """
    error_message = "Something went wrong. "

    if isinstance(e, RateLimitError):
        error_message += "Rate limit exceeded. Please try again in a few moments."
    elif isinstance(e, BadRequestError):
        error_message += (
            "Invalid request parameters. Please check your inputs and try again."
        )
    elif isinstance(e, ServiceUnavailableError):
        error_message += (
            "The service is temporarily unavailable. Please try again later."
        )
    elif isinstance(e, ValueError):
        error_message += f"Invalid value: {str(e)}"
    elif isinstance(e, TimeoutError):
        error_message += "The request timed out. Please try again."
    else:
        error_message += f"Unexpected error: {str(e)}"

    logger.error("Error details: %s: %s", type(e).__name__, str(e))

    await cl.Message(content=error_message).send()
