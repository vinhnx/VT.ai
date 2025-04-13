# Utils API Reference

This page documents the utils module of VT (`vtai/utils/`), which provides various utility functions and classes that support the core functionality of the application.

## Overview

The utils module contains helper functions and classes for configuration management, conversation handling, media processing, error handling, and more. These utilities form the foundation of VT's functionality and are used throughout the application.

## Key Components

### Configuration Management

#### `config.py`

```python
# Initialize the application
def initialize_app(custom_option=False):
    """
    Initialize the VT.ai application.

    Args:
        custom_option: Optional custom configuration flag

    Returns:
        Tuple containing the route layer, assistant ID, and OpenAI clients
    """
    # ...

# Get logger instance
def get_logger():
    """
    Get the application logger.

    Returns:
        Logger instance configured for VT.ai
    """
    # ...
```

#### `constants.py`

```python
# Application constants
APP_NAME = "VT"
DEFAULT_MODEL = "o3-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
# ...
```

#### `llm_providers_config.py`

```python
# Provider configuration
PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "models": ["o1", "o3-mini", "4o"],
        "env_var": "OPENAI_API_KEY",
        "icon": "chatgpt-icon.png",
    },
    # Additional providers...
}

# Model to provider mapping
MODEL_PROVIDER_MAP = {
    "o1": "openai",
    "o3-mini": "openai",
    "4o": "openai",
    # Additional mappings...
}

# Get list of models
def get_available_models():
    """
    Get a list of all available models.

    Returns:
        List of model names
    """
    # ...

# Check if a model is a reasoning model
def is_reasoning_model(model):
    """
    Check if a model is categorized as a reasoning model.

    Args:
        model: Model name to check

    Returns:
        True if the model is a reasoning model, False otherwise
    """
    # ...
```

### Conversation Handlers

#### `conversation_handlers.py`

```python
# Handle standard conversation
async def handle_conversation(message, messages, route_layer):
    """
    Handle a standard conversation message.

    Args:
        message: The user message
        messages: Message history
        route_layer: Semantic routing layer
    """
    # ...

# Handle thinking mode conversation
async def handle_thinking_conversation(message, messages, route_layer):
    """
    Handle a thinking mode conversation message.

    Args:
        message: The user message
        messages: Message history
        route_layer: Semantic routing layer
    """
    # ...

# Configure chat session
async def config_chat_session(settings):
    """
    Configure the chat session with settings.

    Args:
        settings: Chat settings dictionary
    """
    # ...
```

### Media Processors

#### `media_processors.py`

```python
# Handle vision tasks
async def handle_vision(message, messages, client):
    """
    Handle vision analysis for images.

    Args:
        message: The user message
        messages: Message history
        client: LLM client
    """
    # ...

# Handle image generation
async def handle_trigger_async_image_gen(message, messages, client, **kwargs):
    """
    Handle image generation requests.

    Args:
        message: The user message
        messages: Message history
        client: LLM client
        **kwargs: Additional arguments
    """
    # ...

# Handle TTS responses
async def handle_tts_response(text, client):
    """
    Handle text-to-speech response generation.

    Args:
        text: Text to convert to speech
        client: OpenAI client
    """
    # ...
```

### File Handlers

#### `file_handlers.py`

```python
# Process files
async def process_files(elements, client):
    """
    Process uploaded files.

    Args:
        elements: Message elements containing files
        client: OpenAI client

    Returns:
        List of file IDs
    """
    # ...
```

### Error Handlers

#### `error_handlers.py`

```python
# Handle exceptions
async def handle_exception(exception):
    """
    Handle an exception and display appropriate error message.

    Args:
        exception: The exception to handle
    """
    # ...
```

### UI Components

#### `llm_profile_builder.py`

```python
# Build LLM profile
def build_llm_profile(icons_map):
    """
    Build the LLM profile with icons.

    Args:
        icons_map: Mapping of providers to icons
    """
    # ...
```

#### `settings_builder.py`

```python
# Build settings
async def build_settings():
    """
    Build the settings UI.

    Returns:
        Settings components
    """
    # ...
```

### Helper Utilities

#### `dict_to_object.py`

```python
class DictToObject:
    """
    Convert a dictionary to an object with attributes.
    """

    def __init__(self, data):
        """
        Initialize with dictionary data.

        Args:
            data: Dictionary to convert
        """
        # ...
```

#### `user_session_helper.py`

```python
# Get setting value
def get_setting(key, default=None):
    """
    Get a setting value from user session.

    Args:
        key: Setting key
        default: Default value if not found

    Returns:
        Setting value
    """
    # ...

# Check if in assistant profile
def is_in_assistant_profile():
    """
    Check if the current session is using the assistant profile.

    Returns:
        True if in assistant profile, False otherwise
    """
    # ...
```

## Usage Examples

### Configuration Example

```python
from vtai.utils import constants, config
from vtai.utils.llm_providers_config import get_available_models

# Get application logger
logger = config.get_logger()

# Get available models
models = get_available_models()
logger.info("Available models: %s", models)

# Initialize the application
route_layer, assistant_id, openai_client, async_openai_client = config.initialize_app()
```

### Conversation Handling Example

```python
from vtai.utils.conversation_handlers import handle_conversation, handle_thinking_conversation

# Handle standard query
await handle_conversation(message, message_history, route_layer)

# Handle thinking mode query
if "<think>" in message.content:
    await handle_thinking_conversation(message, message_history, route_layer)
```

### Media Processing Example

```python
from vtai.utils.media_processors import handle_vision, handle_tts_response

# Process an image
if message.elements and message.elements[0].type == "image":
    await handle_vision(message, message_history, client)

# Generate speech
await handle_tts_response("Text to convert to speech", openai_client)
```

## Best Practices

When working with the utils module:

1. **Error Handling**:
   - Use the `handle_exception` function for consistent error handling
   - Wrap async operations in try/except blocks

2. **Configuration**:
   - Access constants from the `constants` module
   - Use the provider configuration for model operations

3. **Session Management**:
   - Use the `user_session_helper` functions to access session data
   - Store persistent data in user sessions

## Source Code

For the complete source code of the utils module, see the [GitHub repository](https://github.com/vinhnx/VT.ai/tree/main/vtai/utils).
