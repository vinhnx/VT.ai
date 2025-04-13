"""
Core application constants for VT.

This module contains constants used throughout the application,
including names, identifiers, and feature flags.
"""

from typing import Dict, Final

# Application names and identifiers
APP_NAME: Final[str] = "VT"
MINO_ASSISTANT_NAME: Final[str] = "Mino"

# Assistant tools
ASSISTANT_TOOL_CODE_INTERPRETER: Final[str] = "code_interpreter"
ASSISTANT_TOOL_RETRIEVAL: Final[str] = "retrieval"
ASSISTANT_TOOL_FUNCTION: Final[str] = "function"

# Feature flags
DEBUG_MODE: Final[bool] = False

# Model mappings
MODEL_ALIAS_MAP: Dict[str, str] = {
    # OpenAI models
    "OpenAI - GPT-o1": "o1",
    "OpenAI - GPT-o1 Mini": "o1-mini",
    "OpenAI - GPT-o1 Pro": "o1-pro",
    "OpenAI - GPT-o3 Mini": "o3-mini",
    "OpenAI - GPT-4.5 Preview": "gpt-4.5-preview",
    "OpenAI - GPT-4o": "gpt-4o",
    "OpenAI - GPT-4o Mini": "gpt-4o-mini",
    # Anthropic models
    "Anthropic - Claude 3.7 Sonnet": "claude-3-7-sonnet-20250219",
    "Anthropic - Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
    "Anthropic - Claude 3.5 Haiku": "claude-3-5-haiku-20241022",
    # Google models
    "Google - Gemini 2.0 Pro": "gemini/gemini-2.0-pro",
    "Google - Gemini 2.0 Flash": "gemini/gemini-2.0-flash",
    "Google - Gemini 2.0 Flash Exp": "gemini/gemini-2.0-flash-exp",
    # Common models
    "default_model_name": "gpt-4o-mini",
}
