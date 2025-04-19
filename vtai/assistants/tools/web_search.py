"""
Web search tool definition for OpenAI Assistant API.
"""

# Web search tool definition for the assistant
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for real-time information on any topic. Use when information might be outdated or not in the model's training data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find information about. Make this specific and detailed for better results.",
                },
                "model": {
                    "type": "string",
                    "description": "The model to use for searching (optional, defaults to gpt-4o)",
                    "default": "openai/gpt-4o",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (optional)",
                    "default": 5,
                },
                "search_context_size": {
                    "type": "string",
                    "description": "The size of the search context (low, medium, high)",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
                "include_urls": {
                    "type": "boolean",
                    "description": "Whether to include URLs in the search results",
                    "default": False,
                },
                "summarize_results": {
                    "type": "boolean",
                    "description": "Whether to summarize the results instead of returning raw output",
                    "default": True,
                },
            },
            "required": ["query"],
        },
    },
}
