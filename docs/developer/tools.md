# Assistant Tools

This guide explains the Assistant Tools feature in VT.ai, which provides specialized capabilities beyond standard chat interaction.

## Overview

VT.ai includes an Assistant mode based on OpenAI's Assistants API that provides powerful tools for code interpretation, file handling, and function calling. These tools enable more complex workflows and specialized functionality.

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

### Function Calling

Function Calling allows the assistant to interact with external systems and APIs:

- **Current Status**: Function tools are temporarily disabled in the current version.
- **Planned Features**:
  - Web search integration
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
        # Function tools temporarily disabled
    ],
    "model": "o3"
}
```

### Tool Processing Flow

The processing flow for tools follows this pattern:

1. User submits a query in Assistant mode
2. Query is processed by the appropriate model
3. If the model decides to use a tool, a tool call is generated
4. VT.ai processes the tool call (e.g., executes code, analyzes files)
5. Tool output is returned to the model
6. Model generates final response incorporating tool results

### Processing Tool Calls

Tool calls are processed by specialized handlers:

```python
# Example of code interpreter tool processing
async def process_code_interpreter_tool(
    step_references: Dict[str, cl.Step], step: Any, tool_call: Any
) -> Dict[str, Any]:
    """
    Process code interpreter tool calls.

    Args:
        step_references: Dictionary of step references
        step: The run step
        tool_call: The tool call to process

    Returns:
        Tool output dictionary
    """
    output_value = ""
    if tool_call.code_interpreter.outputs and len(tool_call.code_interpreter.outputs) > 0:
        output_value = tool_call.code_interpreter.outputs[0]

    # Process and display the results in the UI
    await process_tool_call(
        step_references=step_references,
        step=step,
        tool_call=tool_call,
        name=tool_call.type,
        input=tool_call.code_interpreter.input or "# Generating code",
        output=output_value,
        show_input="python",
    )

    return {
        "output": tool_call.code_interpreter.outputs or "",
        "tool_call_id": tool_call.id,
    }
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
- **Output Visualization**: Displays images, charts, and other outputs
- **Interactive Elements**: Allows users to interact with the assistant's outputs

## Extending Assistant Tools

While function tools are temporarily disabled, VT.ai is designed to be extended with custom tools:

1. Define the tool interface in `vtai/tools/`
2. Add the tool to the assistant configuration
3. Implement the tool processing logic
4. Add UI components to display tool outputs

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

3. **General Usage**:
   - Be explicit about what you want the assistant to do
   - Check intermediate results
   - Break complex tasks into smaller steps

*This page is under construction. More detailed information about assistant tools will be added soon.*
