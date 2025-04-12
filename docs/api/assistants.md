# Assistants API Reference

This page documents the assistants module of VT.ai (`vtai/assistants/`), which provides specialized AI assistants with enhanced capabilities.

## Overview

The assistants module implements VT.ai's Assistant mode, which uses OpenAI's Assistants API to provide more advanced features like code interpretation, file processing, and function calling. This module manages the creation and execution of assistant threads and runs.

## Key Components

### Assistant Configuration

The Assistant configuration is defined in the application initialization:

```python
# Create an assistant with tools
async def create_assistant(client, name, instructions, tools=None, model="o3"):
    """
    Create an assistant with specified configuration.

    Args:
        client: OpenAI client
        name: Assistant name
        instructions: System instructions
        tools: List of tool configurations
        model: Model to use

    Returns:
        Assistant object
    """
    # ...
```

### Thread Management

Threads are used to maintain conversation context in Assistant mode:

```python
# Create a new thread
async def create_thread(client):
    """
    Create a new conversation thread.

    Args:
        client: OpenAI client

    Returns:
        Thread object
    """
    # ...

# Add a message to a thread
async def add_message_to_thread(client, thread_id, content, role="user"):
    """
    Add a message to a thread.

    Args:
        client: OpenAI client
        thread_id: ID of the thread
        content: Message content
        role: Message role (user or assistant)

    Returns:
        Message object
    """
    # ...
```

### Run Management

Runs are used to execute assistant operations:

```python
# Create a run
async def create_run(client, thread_id, assistant_id):
    """
    Create a run for assistant processing.

    Args:
        client: OpenAI client
        thread_id: ID of the thread
        assistant_id: ID of the assistant

    Returns:
        Run object
    """
    # ...

# Poll run status
async def poll_run(client, thread_id, run_id):
    """
    Poll the status of a run.

    Args:
        client: OpenAI client
        thread_id: ID of the thread
        run_id: ID of the run

    Returns:
        Updated run object
    """
    # ...
```

## Assistant Tools

### Code Interpreter Tool

The code interpreter tool allows executing Python code:

```python
# Process code interpreter output
async def process_code_interpreter(step_details):
    """
    Process code interpreter output.

    Args:
        step_details: Details of the step

    Returns:
        Processed output
    """
    # ...
```

### Retrieval Tool

The retrieval tool handles information retrieval:

```python
# Process retrieval output
async def process_retrieval(step_details):
    """
    Process retrieval output.

    Args:
        step_details: Details of the step

    Returns:
        Retrieved information
    """
    # ...
```

### Function Calling Tool

The function calling tool enables interaction with external systems:

```python
# Process function call
async def process_function_call(step_details):
    """
    Process function call.

    Args:
        step_details: Details of the step

    Returns:
        Function result
    """
    # ...
```

## Tool Processing

### Tool Call Processing

```python
# Process tool calls
async def process_tool_calls(step_details):
    """
    Process all tool calls in a step.

    Args:
        step_details: Details of the step

    Returns:
        List of tool outputs
    """
    # ...

# Submit tool outputs
async def submit_tool_outputs(client, thread_id, run_id, tool_outputs):
    """
    Submit tool outputs back to the run.

    Args:
        client: OpenAI client
        thread_id: ID of the thread
        run_id: ID of the run
        tool_outputs: List of tool outputs

    Returns:
        Updated run object
    """
    # ...
```

### Message Processing

```python
# Process thread messages
async def process_thread_message(message_references, thread_message, client):
    """
    Process a message from a thread.

    Args:
        message_references: Dictionary of message references
        thread_message: Message from the thread
        client: OpenAI client
    """
    # ...

# Process tool call
async def process_tool_call(step_references, step, tool_call, name, input, output, show_input=None):
    """
    Process and display a tool call.

    Args:
        step_references: Dictionary of step references
        step: The step containing the tool call
        tool_call: The tool call object
        name: Name of the tool
        input: Input to the tool
        output: Output from the tool
        show_input: Format for displaying input
    """
    # ...
```

## Usage Examples

### Creating an Assistant Session

```python
from vtai.utils.assistant_tools import process_thread_message, process_tool_call

# Setup assistant session
async def setup_assistant_session():
    # Create or get assistant ID
    if not assistant_id:
        # Create a new assistant if none exists
        assistant = await create_assistant(
            client=async_openai_client,
            name="VT.ai Code Assistant",
            instructions="You are a helpful code and data analysis assistant",
            tools=[{"type": "code_interpreter"}, {"type": "retrieval"}],
            model="o3"
        )
        assistant_id = assistant.id

    # Create a thread for the conversation
    thread = await async_openai_client.beta.threads.create()
    cl.user_session.set("thread", thread)

    return thread.id
```

### Running an Assistant Query

```python
# Run an assistant query
async def run_assistant_query(thread_id, query):
    # Add the message to the thread
    await async_openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=query,
    )

    # Create a run
    run = await async_openai_client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

    # Track message and step references
    message_references = {}
    step_references = {}
    tool_outputs = []

    # Poll for completion
    while True:
        run = await async_openai_client.beta.threads.runs.retrieve(
            thread_id=thread_id, run_id=run.id
        )

        # Process steps
        steps = await async_openai_client.beta.threads.runs.steps.list(
            thread_id=thread_id, run_id=run.id, order="asc"
        )

        for step in steps.data:
            # Process step details
            # ...

        # Submit tool outputs if needed
        if run.status == "requires_action" and run.required_action.type == "submit_tool_outputs":
            await async_openai_client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )

        # Check if run is complete
        if run.status in ["cancelled", "failed", "completed", "expired"]:
            break

        # Wait before polling again
        await asyncio.sleep(1)
```

## Best Practices

When working with the Assistants API:

1. **Thread Management**:
   - Create a new thread for each conversation
   - Store thread IDs in user sessions
   - Clean up threads when they're no longer needed

2. **Error Handling**:
   - Implement timeouts for long-running operations
   - Handle API failures gracefully
   - Provide user feedback during processing

3. **Tool Processing**:
   - Cache results when appropriate
   - Validate inputs before processing
   - Format outputs for user readability

## Source Code

For the complete source code of the assistants module, see the [GitHub repository](https://github.com/vinhnx/VT.ai/tree/main/vtai/assistants).
