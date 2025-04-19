"""
Safe execution utilities for VT application.

Provides standardized error handling and context management.
"""

import asyncio
import contextlib
from typing import Any, Callable, Optional

import chainlit as cl

from vtai.utils.config import logger
from vtai.utils.error_handlers import handle_exception


@contextlib.asynccontextmanager
async def safe_execution(
    operation_name: str, 
    timeout_message: str = "The operation timed out. Please try again with a simpler query.",
    cancelled_message: str = "The operation was cancelled. Please try again.",
    on_timeout: Optional[Callable] = None,
    on_cancel: Optional[Callable] = None
):
    """
    Context manager for safely executing asynchronous operations with standardized error handling.
    
    Args:
        operation_name: Name of the operation for logging purposes
        timeout_message: Message to display on timeout
        cancelled_message: Message to display when operation is cancelled
        on_timeout: Optional function to call on timeout
        on_cancel: Optional function to call when cancelled
        
    Yields:
        None
    """
    try:
        yield
    except asyncio.TimeoutError:
        logger.error(f"Timeout occurred during {operation_name}")
        if on_timeout:
            await on_timeout()
        else:
            await cl.Message(content=timeout_message).send()
    except asyncio.CancelledError:
        logger.warning(f"{operation_name} was cancelled")
        if on_cancel:
            await on_cancel()
        else:
            await cl.Message(content=cancelled_message).send()
        raise
    except Exception as e:
        logger.error(f"Error in {operation_name}: {e}")
        await handle_exception(e)