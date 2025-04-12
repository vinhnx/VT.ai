# App Module API Reference

This page documents the main application module of VT.ai (`vtai/app.py`), which serves as the entry point and core controller for the application.

## Overview

The app module coordinates the entire VT.ai application, handling user interactions, routing queries, processing responses, and managing the Chainlit web interface. It initializes the application, sets up chat profiles, processes user messages, and manages assistant tools.

## Key Functions

### Application Initialization

```python
route_layer, assistant_id, openai_client, async_openai_client = initialize_app()
```

Initializes the application and returns the routing layer, assistant ID, and OpenAI clients.

### Chat Profile Setup

```python
@cl.set_chat_profiles
async def build_chat_profile(_=None):
    """Define and set available chat profiles."""
    # ...
```

Defines and sets available chat profiles for the Chainlit interface. This function is decorated with `@cl.set_chat_profiles` to register it with the Chainlit framework.

### Chat Session Initialization

```python
@cl.on_chat_start
async def start_chat():
    """Initialize the chat session with settings and system message."""
    # ...
```

Initializes the chat session when a user starts a new conversation. Sets default settings, builds the LLM profile, and configures the session with the selected model.

### Message Processing

```python
@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Handle incoming user messages and route them appropriately.

    Args:
        message: The user message object
    """
    # ...
```

Processes incoming user messages. Determines whether to use assistant mode or standard chat mode, handles file attachments, and routes the message to the appropriate handler.

### Assistant Run Management

```python
@cl.step(name=APP_NAME, type="run")
async def run(thread_id: str, human_query: str, file_ids: Optional[List[str]] = None):
    """
    Run the assistant with the user query and manage the response.

    Args:
        thread_id: Thread ID to interact with
        human_query: User's message
        file_ids: Optional list of file IDs to attach
    """
    # ...
```

Manages assistant runs when using the assistant mode. Creates a thread if necessary, adds the user message, and processes the assistant's response.

### Tool Call Processing

```python
async def process_tool_calls(
    step_details: Any, step_references: Dict[str, cl.Step], step: Any
) -> List[Dict[str, Any]]:
    """
    Process all tool calls from a step.

    Args:
        step_details: The step details object
        step_references: Dictionary of step references
        step: The run step

    Returns:
        List of tool outputs
    """
    # ...
```

Processes tool calls from the assistant, such as code interpreter, retrieval, and function calls.

### Settings Management

```python
@cl.on_settings_update
async def update_settings(settings: Dict[str, Any]) -> None:
    """
    Update user settings based on preferences.

    Args:
        settings: Dictionary of user settings
    """
    # ...
```

Updates user settings when they are changed in the interface. Handles settings for model selection, temperature, top_p, image generation, TTS, and other options.

### TTS Response Handling

```python
@cl.action_callback("speak_chat_response_action")
async def on_speak_chat_response(action: cl.Action) -> None:
    """
    Handle TTS action triggered by the user.

    Args:
        action: The action object containing payload
    """
    # ...
```

Handles text-to-speech actions triggered by the user. Converts text responses to speech using the selected TTS model.

### Configuration Setup

```python
def setup_chainlit_config():
    """
    Sets up a centralized Chainlit configuration directory in ~/.config/vtai/.chainlit
    and creates symbolic links from the current directory to avoid file duplication.
    This process is fully automated and requires no user intervention.

    Returns:
        Path: Path to the centralized chainlit config directory
    """
    # ...
```

Sets up the Chainlit configuration directory and creates necessary symbolic links.

### Main Entry Point

```python
def main():
    """
    Entry point for the VT.ai application when installed via pip.
    This function is called when the 'vtai' command is executed.
    """
    # ...
```

The main entry point for the application. Parses command-line arguments, sets up the environment, and starts the Chainlit server.

## Helper Functions

### Tool Processing Functions

- `process_code_interpreter_tool`: Processes code interpreter tool calls
- `process_function_tool`: Processes function tool calls
- `process_retrieval_tool`: Processes retrieval tool calls

### Run Management

- `create_run_instance`: Creates a run instance for the assistant
- `managed_run_execution`: Context manager for safe run execution

## Usage Examples

### Starting the Application

```python
# Standard startup
main()

# Or if running as the main script
if __name__ == "__main__":
    main()
```

### Custom Configuration

```python
# Example of customizing the initialization
from vtai.app import setup_chainlit_config, initialize_app

# Setup custom configuration
config_dir = setup_chainlit_config()

# Initialize with custom options
route_layer, assistant_id, openai_client, async_openai_client = initialize_app(
    custom_option=True
)
```

## Source Code

For the complete source code of the app module, see the [GitHub repository](https://github.com/vinhnx/VT.ai/blob/main/vtai/app.py).
