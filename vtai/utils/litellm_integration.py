"""
LiteLLM integration module for VT.ai.

This module sets up LiteLLM with Supabase logging callbacks and provides
a unified interface for LLM calls across different providers.
"""

import os
from typing import Any, Dict, List, Optional

import litellm
from litellm import completion
from utils.config import logger
from utils.supabase_logger import setup_litellm_callbacks


class LiteLLMClient:
    """Unified LLM client using LiteLLM with Supabase logging."""

    def __init__(self):
        """Initialize the LiteLLM client with callbacks."""
        self.setup_environment()
        setup_litellm_callbacks()

    def setup_environment(self):
        """Setup environment variables for LiteLLM providers."""
        # Set default API base URLs if not already set
        if not os.environ.get("LITELLM_LOG"):
            os.environ["LITELLM_LOG"] = "DEBUG"

        logger.info("LiteLLM client initialized with environment setup")

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        user: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """
        Make a chat completion request using LiteLLM.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "claude-3-sonnet", "gemini-pro")
            messages: List of message dictionaries
            user: User identifier for logging
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional arguments for litellm.completion

        Returns:
            Completion response from LiteLLM
        """
        try:
            response = completion(
                model=model,
                messages=messages,
                user=user,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs,
            )
            return response
        except Exception as e:
            logger.error(
                "Error in LiteLLM completion: %s: %s", type(e).__name__, str(e)
            )
            raise

    def image_generation(
        self,
        prompt: str,
        model: str = "dall-e-3",
        user: Optional[str] = None,
        size: str = "1024x1024",
        quality: str = "standard",
        **kwargs,
    ) -> Any:
        """
        Generate an image using LiteLLM.

        Args:
            prompt: Image generation prompt
            model: Image model to use
            user: User identifier for logging
            size: Image size
            quality: Image quality
            **kwargs: Additional arguments

        Returns:
            Image generation response
        """
        try:
            response = litellm.image_generation(
                prompt=prompt,
                model=model,
                user=user,
                size=size,
                quality=quality,
                **kwargs,
            )
            return response
        except Exception as e:
            logger.error(
                "Error in LiteLLM image generation: %s: %s", type(e).__name__, str(e)
            )
            raise


# Global client instance
llm_client = LiteLLMClient()


def get_llm_client() -> LiteLLMClient:
    """Get the global LiteLLM client instance."""
    return llm_client


def quick_completion(
    model: str, messages: List[Dict[str, str]], user: Optional[str] = None, **kwargs
) -> Any:
    """Quick completion function for convenience."""
    return llm_client.chat_completion(
        model=model, messages=messages, user=user, **kwargs
    )
