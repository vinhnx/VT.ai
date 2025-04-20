"""
MCP (Model Context Protocol) server configuration for VT.ai.

This module configures the available tools for the MCP server and contains configuration settings for the MCP integration.
"""

import os
from typing import Dict, List, Optional

# MCP Server Configuration
MCP_HOST = "localhost"
MCP_PORT = 9393

# MCP Model Mappings
# Maps standardized model names to provider-specific model names
MCP_MODEL_MAP = {
    # OpenAI models
    "gpt-4": "o1",
    "gpt-4-turbo": "o1",
    "gpt-4o-mini": "o3-mini",
    # Anthropic models
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-5-haiku-20241022",
    # Google models
    "gemini-pro": "gemini/gemini-2.0-pro",
    "gemini-flash": "gemini/gemini-2.0-flash",
    # Mistral models
    "mistral-large": "mistral/mistral-large-latest",
    "mistral-small": "mistral/mistral-small-latest",
    # Meta models
    "llama-3-70b": "meta-llama/llama-3-70b-instruct",
    "llama-3-8b": "meta-llama/llama-3-8b-instruct",
    # Default model alias - used when no specific model is requested
    "default": "o3-mini",
}

# Default model to use when none is specified
MCP_DEFAULT_MODEL = "o3-mini"

# Timeout in seconds for MCP API requests
MCP_API_TIMEOUT = 120

# Stream chunk size for streaming responses (in characters)
MCP_STREAM_CHUNK_SIZE = 4

# Whether to enable debug mode for MCP logging
MCP_DEBUG = False

# Additional model parameters
MCP_DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 4000,
}

# Maximum message context size (in characters)
# to prevent overloading the MCP server with large requests
MCP_MAX_CONTEXT_SIZE = 1_000_000

# Example tool configurations
WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location to get weather for (city, country)",
            }
        },
        "required": ["location"],
    },
}

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for information",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

IMAGE_GENERATION_TOOL = {
    "name": "generate_image",
    "description": "Generate an image based on a text prompt",
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Text description of the image to generate",
            },
            "style": {
                "type": "string",
                "description": "Style of the image (e.g., photo, cartoon, abstract)",
                "enum": ["photo", "cartoon", "abstract", "painting"],
                "default": "photo",
            },
        },
        "required": ["prompt"],
    },
}

# List of available tools for the MCP server
AVAILABLE_TOOLS = [WEATHER_TOOL, WEB_SEARCH_TOOL, IMAGE_GENERATION_TOOL]
