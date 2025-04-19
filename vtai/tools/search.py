"""
Web search tools using liteLLM and Tavily for VT.ai
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from litellm import completion
from pydantic import BaseModel, Field

# Attempt to import TavilyClient at the top level
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

logger = logging.getLogger(__name__)


class WebSearchOptions(BaseModel):
    """Options for web search"""

    search_context_size: str = Field(
        default="medium",
        description="The size of the search context to use. Options: 'low', 'medium', 'high'",
    )
    include_urls: bool = Field(
        default=False, description="Whether to include URLs in the response"
    )


class WebSearchParameters(BaseModel):
    """Parameters for web search"""

    query: str = Field(..., description="The query to search for")
    model: str = Field(
        default="openai/gpt-4o", description="The model to use for search"
    )
    max_results: Optional[int] = Field(
        default=None, description="Maximum number of search results to return"
    )
    search_options: Optional[WebSearchOptions] = Field(
        default=None, description="Options for web search"
    )
    use_tavily: bool = Field(
        default=False, description="Whether to use Tavily instead of LiteLLM for search"
    )
    tavily_api_key: Optional[str] = Field(
        default=None, description="Tavily API key (optional)"
    )


class WebSearchTool:
    """Tool for performing web searches using liteLLM or Tavily"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
    ):
        """
        Initialize the web search tool

        Args:
            api_key: The API key to use (optional if using a proxy with auth)
            base_url: Base URL for the API (optional if using direct OpenAI access)
            tavily_api_key: Tavily API key (optional)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.tavily_api_key = tavily_api_key
        logger.info("Initialized WebSearchTool")

    async def search_with_tavily(self, params: WebSearchParameters) -> Dict[str, Any]:
        """
        Perform a web search using Tavily API

        Args:
            params: The search parameters

        Returns:
            The search results
        """
        # Check if Tavily is available before proceeding
        if not TAVILY_AVAILABLE:
            logger.error(
                "Tavily package is not installed. Install with: uv pip install tavily-python"
            )
            return {
                "status": "error",
                "error": "Tavily package is not installed. Install with: uv pip install tavily-python",
                "response": f"I couldn't search for '{params.query}' because the Tavily package is not installed.",
            }

        try:
            # Get API key from params or use the one from initialization
            tavily_api_key = params.tavily_api_key or self.tavily_api_key

            if not tavily_api_key:
                raise ValueError("Tavily API key is required for Tavily search")

            # Initialize Tavily client
            tavily_client = TavilyClient(api_key=tavily_api_key)

            # Set up search parameters
            search_kwargs = {}

            # Add max results if specified
            if params.max_results:
                search_kwargs["max_results"] = params.max_results

            # Add search depth based on context size if specified
            if params.search_options and params.search_options.search_context_size:
                # Map our context size to Tavily's search_depth
                depth_map = {"low": "basic", "medium": "basic", "high": "advanced"}
                search_kwargs["search_depth"] = depth_map.get(
                    params.search_options.search_context_size, "basic"
                )

            # Include URLs if specified
            include_urls = False
            if params.search_options and params.search_options.include_urls:
                include_urls = params.search_options.include_urls

            # Perform the search
            logger.info(f"Performing Tavily search for query: {params.query}")
            response = tavily_client.search(params.query, **search_kwargs)

            # Log the raw Tavily API response
            logger.debug(f"Raw Tavily API response: {response}") # Changed print to logger.debug

            # Process the response
            content = response.get("answer") # Get answer, could be None
            results = response.get("results", [])
            sources = []

            # If no direct answer, synthesize from results
            if not content and results:
                logger.info("No direct answer from Tavily, synthesizing from results.")
                summary_parts = []
                for res in results:
                    title = res.get('title', 'No Title')
                    snippet = res.get('content', 'No Content')
                    # Avoid adding results with failed content retrieval
                    if snippet and snippet.lower() != 'failed':
                        summary_parts.append(f"**{title}**: {snippet}")
                if summary_parts:
                    content = "\n\n".join(summary_parts)
                else:
                    logger.warning("Tavily results had no usable content for synthesis.")
                    content = f"I found some related titles for '{params.query}' but couldn't extract detailed content."
            elif not content and not results:
                 logger.warning("Tavily returned no answer and no results.")
                 # Content will be set in the result dictionary creation below

            # Extract sources if available and include_urls is True
            if include_urls and results:
                for idx, res_data in enumerate(results, 1):
                    if res_data.get("url"): # Only add sources with URLs
                        source = {
                            "title": res_data.get("title", f"Result {idx}"),
                            "url": res_data.get("url"),
                        }
                        sources.append(source)

            # Create result dictionary
            final_response_text = content
            if not final_response_text: # Set fallback if content is still empty
                final_response_text = f"I searched for information about '{params.query}' but couldn't find relevant results."

            result_dict = {
                "status": "success",
                "response": final_response_text,
                "model": "Tavily Search API",
            }

            # Add sources if available
            if sources:
                result_dict["sources_json"] = json.dumps({"sources": sources})

            return result_dict

        except Exception as e:
            logger.error(f"Error performing Tavily search: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "response": f"I encountered an error while searching for '{params.query}' using Tavily: {str(e)}",
            }

    async def search(self, params: WebSearchParameters) -> Dict[str, Any]:
        """
        Perform a web search using liteLLM or Tavily

        Args:
            params: The search parameters

        Returns:
            The search results
        """
        # Use Tavily if specified
        if params.use_tavily or (
            hasattr(self, "tavily_api_key") and self.tavily_api_key
        ):
            return await self.search_with_tavily(params)

        # Otherwise use liteLLM
        logger.info(f"Performing web search for query: {params.query}")

        try:
            # Set up messages for the completion
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate information using web search when needed.",
                },
                {
                    "role": "user",
                    "content": f"Please search the web for: {params.query}",
                },
            ]

            # Define web search tool
            web_search_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for real-time information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ]

            # Add search options if specified
            if params.search_options:
                if params.search_options.search_context_size:
                    search_context_property = {
                        "search_context_size": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "The size of the search context",
                        }
                    }
                    web_search_tools[0]["function"]["parameters"]["properties"].update(
                        search_context_property
                    )

                if params.search_options.include_urls:
                    include_urls_property = {
                        "include_urls": {
                            "type": "boolean",
                            "description": "Whether to include URLs in the search results",
                        }
                    }
                    web_search_tools[0]["function"]["parameters"]["properties"].update(
                        include_urls_property
                    )

            # Prepare tool parameters
            tool_choice = {"type": "function", "function": {"name": "web_search"}}

            # Make completion call with tool choice
            response = completion(
                model=params.model,
                messages=messages,
                tools=web_search_tools,
                tool_choice=tool_choice,
                tool_use_system_prompt="Always use the web_search function for queries requiring current information.",
                api_key=self.api_key,
                api_base=self.base_url,
            )

            # Extract content from response - key change here
            content = response.choices[0].message.content

            # If content is None or empty, try to get tool calls
            if not content and hasattr(response.choices[0].message, "tool_calls"):
                # Process tool calls if available
                tool_calls = response.choices[0].message.tool_calls
                sources = []

                # Default response if we can't extract anything
                content = f"I searched for information about '{params.query}' but couldn't find relevant results."

                for tool_call in tool_calls:
                    if (
                        hasattr(tool_call, "function")
                        and tool_call.function.name == "web_search"
                    ):
                        try:
                            # Parse function arguments
                            arguments = json.loads(tool_call.function.arguments)

                            # If there's content in the arguments, use it
                            if "content" in arguments:
                                content = arguments["content"]

                            # Extract sources if available
                            if "sources" in arguments:
                                sources = arguments["sources"]
                        except Exception as e:
                            logger.error(f"Error extracting data from tool call: {e}")

                # Create result with extracted data
                result = {
                    "status": "success",
                    "response": content or "No relevant information found.",
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                }

                # Add sources if available
                if sources:
                    result["sources_json"] = json.dumps({"sources": sources})

                return result

            # If there's direct content in the response, use it
            result = {
                "status": "success",
                "response": content or "No relevant information found.",
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

            return result

        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "response": f"I encountered an error while searching for '{params.query}': {str(e)}",
            }

    # Synchronous version
    def search_sync(self, params: WebSearchParameters) -> Dict[str, Any]:
        """
        Perform a synchronous web search (for use outside async contexts)

        Args:
            params: The search parameters

        Returns:
            The search results
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.search(params))
