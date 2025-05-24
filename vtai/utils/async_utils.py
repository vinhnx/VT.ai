import functools
import asyncio
from typing import Callable, Any, Awaitable


def make_async(func: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """
    Convert a synchronous function to an asynchronous function that runs in a thread.

    Args:
            func: The synchronous function to run in a separate thread.

    Returns:
            Coroutine: An async function that runs the original function in a thread.
    """

    @functools.wraps(func)
    async def async_func(*args, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )

    return async_func
