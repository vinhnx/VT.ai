from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AppChatProfileType(str, Enum):
    """
    Enumeration of available chat profile types in the application.

    Attributes:
        CHAT: Regular chat interface with LLM
        ASSISTANT: Advanced assistant with additional capabilities
    """
    CHAT = "Chat"
    ASSISTANT = "Assistant"


class AppChatProfileModel(BaseModel):
    """
    Model representing a chat profile configuration.

    Attributes:
        title: Display name of the chat profile
        description: Markdown-compatible description of the profile capabilities
        icon: Optional path to the profile icon
        is_default: Whether this profile should be the default selection
    """
    title: str = Field(..., description="Display name of the profile")
    description: str = Field(..., description="Profile description in markdown format")
    icon: Optional[str] = Field(None, description="Path to profile icon image")
    is_default: bool = Field(False, description="Whether this profile is the default")
