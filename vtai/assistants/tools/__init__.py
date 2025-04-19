"""
Assistant tools configuration for VT.ai.

Defines available tools for OpenAI Assistant API integration.
"""

# Import individual tool definitions
from vtai.assistants.tools.web_search import WEB_SEARCH_TOOL

# List of available assistant tools
ASSISTANT_TOOLS = [
    {"type": "code_interpreter"},  # Built-in code interpreter tool
    {"type": "retrieval"},  # Built-in retrieval tool
    WEB_SEARCH_TOOL,  # Custom web search tool
]

# Export the available tools list
__all__ = ["ASSISTANT_TOOLS"]
