# Model Context Protocol (MCP) Integration

VT.ai includes built-in support for the Model Context Protocol (MCP), a standardized way to interact with language models across different providers. This document explains how to use the MCP integration in your VT.ai applications.

## What is MCP?

The Model Context Protocol (MCP) is a standardized protocol for AI model providers and clients to exchange information. It helps standardize how models receive inputs and return outputs, making it easier to swap models or use multiple models in a system.

MCP provides several benefits:

- **Model Interchangeability**: Switch easily between models without changing your code
- **Standardized Interface**: Consistent API for all language models
- **Improved Stability**: Reduced dependency on provider-specific API changes
- **Unified Response Format**: Consistent response structure regardless of provider
- **Enhanced Reliability**: Built-in error handling and fallback mechanisms

## How MCP Works in VT.ai

When you start VT.ai, an MCP server is automatically launched in the background. This server acts as a middleware between your application and various language models, providing a standardized interface.

The MCP server:

1. Accepts requests in a standardized format
2. Maps standardized model names to provider-specific models
3. Forwards the request to the appropriate provider via LiteLLM
4. Returns responses in a standardized format
5. Handles errors and retries gracefully

```
┌─────────────────┐     ┌───────────────┐     ┌────────────────┐
│ VT.ai           │     │ MCP Server    │     │ LiteLLM        │
│ Application     │────▶│ Middleware    │────▶│ Client         │
└─────────────────┘     └───────────────┘     └────────────────┘
        │                       │                     │
        ▼                       ▼                     ▼
┌─────────────────┐     ┌───────────────┐     ┌────────────────┐
│ User            │     │ Model         │     │ AI Provider    │
│ Interface       │     │ Mapping       │     │ APIs           │
└─────────────────┘     └───────────────┘     └────────────────┘
```

## The Standalone MCP Server

VT.ai includes a standalone MCP server in the `mcp_server/` directory that can be run independently of the main application.

### Features of the Standalone Server

- **Independent Operation**: Can run separately from the main VT.ai application
- **API Standardization**: Uses LiteLLM to standardize interactions with different LLM providers
- **Flexible Configuration**: Configure through environment variables or command-line arguments
- **Simple HTTP API**: Follows the Model Context Protocol specification
- **Cross-Application Usage**: Can be used by multiple applications simultaneously

### Running the Standalone Server

You can run the MCP server directly:

```bash
# From the project root
./scripts/run_mcp_server.sh

# Or manually
cd mcp_server
python server.py
```

By default, the server runs on `localhost:9393`, but you can customize this:

```bash
# Custom host and port
export MCP_HOST="your-host"
export MCP_PORT="your-port"
./scripts/run_mcp_server.sh
```

## Using MCP in Your Applications

### Basic Usage

You can use MCP in your VT.ai applications by importing the necessary components:

```python
from utils.mcp_integration import create_mcp_completion, initialize_mcp

# Initialize MCP configuration
mcp_config = initialize_mcp()

# Create a completion
async def example_completion():
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]

    response = await create_mcp_completion(
        messages=messages,
        model="gpt-4o-mini",  # This gets mapped to the actual provider model
        temperature=0.7
    )

    content = response.choices[0].message.content
    print(content)
```

### Streaming Responses

MCP supports streaming responses for a better user experience:

```python
from utils.mcp_integration import create_mcp_completion

async def example_streaming():
    messages = [
        {"role": "user", "content": "Write a short poem about AI"}
    ]

    # Define a callback for streaming
    def stream_callback(token):
        print(token, end="", flush=True)

    # Stream the response
    stream = await create_mcp_completion(
        messages=messages,
        model="gpt-4",
        temperature=0.7,
        stream=True,
        stream_callback=stream_callback
    )

    # Process the stream
    async for chunk in stream:
        # The callback already handles printing
        pass
```

### Using with Chainlit

VT.ai includes a dedicated handler for using MCP with Chainlit:

```python
import chainlit as cl
from utils.mcp_integration import ChainlitMCPHandler, initialize_mcp

# Initialize MCP handler
mcp_handler = ChainlitMCPHandler()

@cl.on_message
async def on_message(message: cl.Message):
    # Get message history
    message_history = cl.user_session.get("message_history", [])

    # Add user message
    message_history.append({"role": "user", "content": message.content})

    # Create response message
    response_message = cl.Message(content="")
    await response_message.send()

    # Handle with MCP
    response_text = await mcp_handler.handle_message(
        message_history=message_history,
        current_message=response_message,
        model="gpt-4o-mini",
        temperature=0.7
    )

    # Add to history
    message_history.append({"role": "assistant", "content": response_text})
    cl.user_session.set("message_history", message_history)
```

## Configuring MCP

You can configure MCP by modifying the settings in `vtai/utils/mcp_config.py` or by setting environment variables:

- `MCP_HOST`: Host address for the MCP server (default: "localhost")
- `MCP_PORT`: Port for the MCP server (default: 9393)
- `MCP_DEFAULT_MODEL`: Default model to use when none is specified (default: "o3-mini")
- `MCP_TIMEOUT`: Request timeout in seconds (default: 60)
- `MCP_MAX_RETRIES`: Maximum number of retry attempts (default: 3)

Model mappings are defined in `MCP_MODEL_MAP` in the config file.

## Example Application

VT.ai includes a complete demo application that showcases MCP integration with Chainlit. You can run it with:

```bash
chainlit run examples/mcp_demo.py
```

This demo shows:

- Model switching without changing application code
- Streaming responses
- Temperature adjustment
- Token count display
- Error handling and retries

## Advanced Usage

### Custom Model Mappings

You can create custom model mappings:

```python
from utils.mcp_integration import initialize_mcp

custom_model_map = {
    "my-fast-model": "o3-mini",
    "my-smart-model": "o1",
    "my-creative-model": "claude-3-opus-20240229"
}

mcp_config = initialize_mcp(model_map=custom_model_map)
```

### Direct API Calls

You can call the MCP API directly:

```python
from utils.mcp_integration import call_mcp_api

async def direct_call():
    messages = [
        {"role": "user", "content": "Hello, world!"}
    ]

    response = await call_mcp_api(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0.7
    )

    print(response)
```

### Multimodal Support

MCP supports multimodal inputs including images:

```python
from utils.mcp_integration import create_mcp_completion
import base64

async def image_analysis():
    # Read image file and convert to base64
    with open("image.jpg", "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Create multimodal message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
    ]

    # Send to vision-capable model
    response = await create_mcp_completion(
        messages=messages,
        model="gpt-4o",  # Must be a vision-capable model
        temperature=0.5
    )

    print(response.choices[0].message.content)
```

## Performance Optimizations

### Connection Pooling

The MCP server uses connection pooling to improve performance when making multiple requests:

```python
# Configure connection pooling
import aiohttp
import asyncio

async def batch_requests():
    # Create a shared session
    async with aiohttp.ClientSession() as session:
        # Create 10 completion requests
        tasks = [
            call_mcp_api(
                messages=[{"role": "user", "content": f"Question {i}"}],
                model="o3-mini",
                session=session  # Pass the session to reuse connections
            )
            for i in range(10)
        ]

        # Run all requests concurrently
        results = await asyncio.gather(*tasks)
```

### Local Caching

You can implement caching to reduce duplicate requests:

```python
import hashlib
import json
import functools

# Simple cache for MCP responses
response_cache = {}

async def cached_mcp_call(messages, model, temperature=0.7):
    # Create a cache key from the input parameters
    key_data = {
        "messages": messages,
        "model": model,
        "temperature": temperature
    }
    cache_key = hashlib.md5(json.dumps(key_data).encode()).hexdigest()

    # Check cache
    if cache_key in response_cache:
        print("Using cached response")
        return response_cache[cache_key]

    # Make actual API call
    response = await call_mcp_api(
        messages=messages,
        model=model,
        temperature=temperature
    )

    # Cache the response
    response_cache[cache_key] = response
    return response
```

## Troubleshooting

- If you encounter errors connecting to the MCP server, check that it's running by looking for the log message "Started MCP server on localhost:9393"
- If a model isn't working, verify that you've set the appropriate API key for that provider
- For model mapping issues, check the `MCP_MODEL_MAP` in `mcp_config.py`
- If the server is unresponsive, try restarting it with `./scripts/run_mcp_server.sh`
- For persistent issues, check the server logs in the terminal where it's running

## Extending the MCP Server

You can extend the MCP server by:

1. Adding new model mappings in `mcp_config.py`
2. Implementing custom response processors
3. Creating middleware for request/response modification
4. Adding custom error handling logic

For more advanced customization, you can modify `mcp_server/server.py` directly.

---

For more information on the Model Context Protocol, visit [the MCP documentation](https://github.com/lostinthestack/mcp).
