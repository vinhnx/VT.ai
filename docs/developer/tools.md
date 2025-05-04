# Assistant Tools

This guide explains the Assistant Tools feature in VT.ai, which provides specialized capabilities beyond standard chat interaction.

## Overview

VT.ai includes an Assistant mode based on OpenAI's Assistants API that provides powerful tools for code interpretation, file handling, web search, and function calling. These tools enable complex workflows and specialized functionality.

## Available Tools

### Code Interpreter

The Code Interpreter tool allows executing Python code directly within the chat interface:

- **Features**:
  - Run Python code in a sandboxed environment
  - Create charts and visualizations
  - Perform data analysis
  - Execute mathematical computations
  - Generate plots and figures

- **Usage**:

  ```
  # Sample code execution in Assistant mode
  import matplotlib.pyplot as plt
  import numpy as np

  x = np.linspace(0, 10, 100)
  y = np.sin(x)

  plt.plot(x, y)
  plt.title("Sine Wave")
  plt.xlabel("X")
  plt.ylabel("sin(x)")
  plt.show()
  ```

### File Processing

The File Processing tool enables working with uploaded files:

- **Supported File Types**:
  - Text files (`.txt`, `.md`, etc.)
  - CSV and spreadsheets (`.csv`, `.xlsx`)
  - Images (`.jpg`, `.png`, etc.)
  - PDFs (`.pdf`)
  - Code files (`.py`, `.js`, etc.)

- **Capabilities**:
  - Extract text from documents
  - Analyze images
  - Process structured data
  - Generate insights from file content

### Web Search

The Web Search tool enables querying the web for current information:

- **Features**:
  - Real-time web search for up-to-date information
  - Smart summarization of search results
  - Source attribution and URL inclusion
  - Customizable search parameters

- **Implementation**:

  The WebSearchTool is implemented in `vtai/tools/search.py` and integrates with the Tavily API for enhanced search capabilities. It provides:

  - Context-aware search
  - Domain filtering (include/exclude specific sites)
  - Result summarization
  - Customizable result count

- **Usage Example**:

  ```python
  from tools import WebSearchTool, WebSearchOptions

  # Initialize the search tool with API keys
  search_tool = WebSearchTool(
      api_key="your-openai-key",
      tavily_api_key="your-tavily-key"
  )

  # Configure search options
  options = WebSearchOptions(
      search_context_size="medium",  # Options: small, medium, large
      include_urls=True,             # Include source URLs
      summarize_results=True         # Provide an AI-generated summary
  )

  # Execute search with options
  results = await search_tool.search(
      query="latest developments in AI",
      model="openai/gpt-4o",
      search_options=options,
      max_results=5
  )
  ```

- **Return Format**:

  The search tool returns structured results including:

  ```json
  {
      "search_results": [
          {"title": "Article Title", "url": "https://example.com/article", "content": "Snippet of content..."},
          // Additional results...
      ],
      "summary": "An AI-generated summary of the search results",
      "query": "Original search query"
  }
  ```

### Function Calling

Function Calling allows the assistant to interact with external systems and APIs:

- **Current Status**: Basic function tools are implemented with a focus on web search integration.
- **Available Functions**:
  - Web search (integrated with Tavily)
  - Context-aware information retrieval
  - Search summarization
- **Planned Functions**:
  - External API calls
  - Database interactions
  - System operations

## Implementation Details

### Assistant Configuration

The Assistant configuration is defined in the codebase:

```python
# Assistant configuration pseudocode
assistant = {
    "name": "VT.ai Code Assistant",
    "description": "A helpful code and data analysis assistant",
    "instructions": "You are a helpful assistant that can execute code...",
    "tools": [
        {"type": "code_interpreter"},
        {"type": "retrieval"},
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "model": {"type": "string", "description": "Model for processing"},
                        "max_results": {"type": "integer", "description": "Maximum results"}
                    },
                    "required": ["query"]
                }
            }
        }
    ],
    "model": "o3"
}
```

### Tool Processing Flow

The processing flow for tools follows this pattern:

1. User submits a query in Assistant mode
2. Query is processed by the appropriate model
3. If the model decides to use a tool, a tool call is generated
4. VT.ai processes the tool call (e.g., executes code, searches the web)
5. Tool output is returned to the model
6. Model generates final response incorporating tool results

### Processing Tool Calls

Tool calls are processed by specialized handlers:

```python
# Example of web search tool processing (simplified)
async def process_function_tool(
    step_references: Dict[str, cl.Step], step: Any, tool_call: Any
) -> Dict[str, Any]:
    """Process function tool calls like web search."""
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    if function_name == "web_search":
        # Initialize the web search tool
        web_search_tool = WebSearchTool(
            api_key=openai_api_key, tavily_api_key=tavily_api_key
        )

        # Extract search parameters
        query = function_args.get("query", "")
        model = function_args.get("model", "openai/gpt-4o")
        max_results = function_args.get("max_results", None)

        # Build search options
        search_options = WebSearchOptions(
            search_context_size="medium",
            include_urls=True,
            summarize_results=True,
        )

        # Execute search
        search_result = await web_search_tool.search(
            query=query,
            model=model,
            search_options=search_options,
            max_results=max_results,
        )

        # Process and return results
        return {
            "output": search_result,
            "tool_call_id": tool_call.id,
        }

    # Handle other function tools...
```

## Thread Management

Assistant mode uses threads to maintain conversation context:

- **Thread Creation**: A new thread is created at the start of each Assistant mode session
- **Message Storage**: Messages are stored in the thread for context
- **Run Management**: Each user query creates a "run" instance that processes the query
- **Step Tracking**: Individual steps within a run are tracked and displayed

## User Interface

The Assistant Tools interface in VT.ai provides:

- **Step Visualization**: Shows each step of the assistant's work
- **Code Displays**: Properly formatted code blocks with syntax highlighting
- **Search Results Display**: Formatted presentation of web search results
- **Output Visualization**: Displays images, charts, and other outputs
- **Interactive Elements**: Allows users to interact with the assistant's outputs

## Extending Assistant Tools

VT.ai is designed to be extended with custom tools:

1. Define the tool interface in `vtai/tools/`
2. Add the tool to the assistant configuration
3. Implement the tool processing logic
4. Add UI components to display tool outputs

### Creating a Custom Tool

Here's an example of how to create a new tool:

```python
# In vtai/tools/my_custom_tool.py
from typing import Dict, Any, Optional

class MyCustomTool:
    """A custom tool implementation for VT.ai."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the custom tool."""
        self.api_key = api_key

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input and generate a result.

        Args:
            input_data: The input parameters for the tool

        Returns:
            Dict containing the processing results
        """
        # Tool implementation
        result = {"status": "success", "data": "Custom tool output"}
        return result
```

Then register it in the assistant configuration and add a handler in the function tool processor.

See the [Extending VT.ai](extending.md) guide for more details.

## Best Practices

When using Assistant Tools:

1. **Code Interpreter**:
   - Keep code snippets focused on a single task
   - For data analysis, provide clear column descriptions
   - Use visualization when appropriate

2. **File Handling**:
   - Provide context about uploaded files
   - Ask specific questions about file content
   - Upload files in appropriate formats

3. **Web Search**:
   - Use specific, focused search queries
   - Specify the number of results when quality is important
   - Ask follow-up questions about search results for clarification

4. **General Usage**:
   - Be explicit about what you want the assistant to do
   - Check intermediate results
   - Break complex tasks into smaller steps

## Limitations and Considerations

- **API Rate Limits**: Web search and other API-based tools may have rate limits
- **Result Freshness**: Web search results reflect the state of the web at the time of the search
- **Processing Time**: Complex tool operations may take longer to complete
- **Tool Selection**: The model decides which tool to use based on the query; you cannot directly specify a tool
- **Tool Chaining**: Tools can be chained together (e.g., search → code interpretation) for more complex workflows

## Future Enhancements

Planned enhancements for Assistant Tools include:

- **Enhanced Tool Discovery**: More intuitive UI for understanding available tools
- **Custom Tool Registry**: User-defined tools with a simplified registration process
- **Tool Permissions**: Granular control over which tools can be used
- **Improved Visualization**: Better display of complex tool outputs
- **Additional Integrations**: More built-in tools for common services and APIs
