"""
Core application constants for VT.ai.

This module contains constants used throughout the application,
including names, identifiers, and feature flags.
"""

from typing import Final

# Application names and identifiers
APP_NAME: Final[str] = "VT.ai"
MINO_ASSISTANT_NAME: Final[str] = "Mino"

# Assistant tools
ASSISTANT_TOOL_CODE_INTERPRETER: Final[str] = "code_interpreter"
ASSISTANT_TOOL_RETRIEVAL: Final[str] = "retrieval"
ASSISTANT_TOOL_FUNCTION: Final[str] = "function"

# Feature flags
DEBUG_MODE: Final[bool] = False
