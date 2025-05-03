# Standalone MCP Server for VT.ai

This directory contains a standalone implementation of the Model Context Protocol (MCP) server for VT.ai. The MCP server provides a standardized interface for interacting with various Language Models (LLMs) through a consistent API.

## Features

- Standalone server that can run independently of the main VT.ai application
- Uses LiteLLM to standardize interactions with different LLM providers
- Configurable through environment variables or command-line arguments
- Simple HTTP API that follows the Model Context Protocol specification

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or uv (recommended) package manager

### Installation

Install the required dependencies:

```bash
uv pip install -r requirements.txt
```

### Running the Server

You can run the server directly from the `mcp_server` directory:

```bash
python server.py
```

Or use the provided shell script from the project root:

```bash
./scripts/run_mcp_server.sh
```

## Configuration

The server can be configured using either environment variables or command-line arguments:

### Environment Variables

- `MCP_HOST`: The host to bind the server to (default: "localhost")
- `MCP_PORT`: The port to bind the server to (default: 9393)
- `MCP_DEFAULT_MODEL`: The default model to use if none is specified (default: "gpt-4o-mini")
- `MCP_MODEL_MAP`: A JSON string mapping model names to provider-specific names (default: "{}")

### Command-Line Arguments

- `--host`: The host to bind the server to
- `--port`: The port to bind the server to
- `--default-model`: The default model to use
- `--model-map`: A JSON string mapping model names

Command-line arguments take precedence over environment variables.

## Usage

The MCP server exposes an HTTP API that follows the [Model Context Protocol](https://github.com/llm-mcp/model-context-protocol) specification. Clients can send requests to the server to interact with LLMs in a standardized way.

### Example Client

Here's a simple example of how to use the server from Python:

```python
import aiohttp
import asyncio
import json

async def example_request():
    async with aiohttp.ClientSession() as session:
        url = "http://localhost:9393/completion"

        payload = {
            "data": [
                {"text": "Hello, how are you?", "metadata": {"role": "user"}}
            ],
            "options": {
                "model": "gpt-4o-mini",
                "temperature": 0.7
            }
        }

        async with session.post(url, json=payload) as response:
            result = await response.json()
            print(result)

asyncio.run(example_request())
```

## API Reference

### `/completion`

The main endpoint for getting completions from LLMs.

**Method**: POST

**Request Body**:

```json
{
  "data": [
    {"text": "Message content", "metadata": {"role": "user"}}
  ],
  "options": {
    "model": "model-name",
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

**Response**:

```json
{
  "content": "Response content"
}
```

## Extending

You can extend the server by modifying the `server.py` file to add new handlers for different types of requests or to support additional LLM providers.

## License

See the project's main LICENSE file for details.
