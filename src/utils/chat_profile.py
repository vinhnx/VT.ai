from enum import Enum

from pydantic import BaseModel


class AppChatProfileType(str, Enum):
    CHAT = "Chat"
    ASSISTANT = "Assistant"


class AppChatProfileModel(BaseModel):
    title: str
    description: str
