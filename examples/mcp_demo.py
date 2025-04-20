"""
Example demonstrating how to use MCP with Chainlit in VT.ai.

This module shows how to use the Model Context Protocol integration in a Chainlit app,
showcasing streaming capabilities and model switching.
"""

import asyncio
import os
from typing import Any, Dict, List

import chainlit as cl

# Import MCP integration components
from vtai.utils.mcp_integration import (
    ChainlitMCPHandler,
    MCPConfig,
    call_mcp_api,
    get_mcp_url,
    initialize_mcp,
)

# Initialize MCP handler
mcp_handler = None
available_models = []

# Set MCP port explicitly to 9393
os.environ["MCP_PORT"] = "9393"


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session with MCP integration."""
    global mcp_handler, available_models

    # Set up Chainlit with MCP
    cl.user_session.set("message_history", [])
    cl.user_session.set("streaming", True)  # Default to streaming enabled
    cl.user_session.set("model", "gpt-4o-mini")  # Default model
    cl.user_session.set("temperature", 0.7)  # Default temperature

    # Initialize MCP handler with default configuration
    mcp_config = initialize_mcp()
    mcp_handler = ChainlitMCPHandler()  # Remove the mcp_config parameter

    # Store available models for actions
    available_models = list(mcp_config.model_map.keys())

    # Display a message about MCP
    mcp_url = get_mcp_url(mcp_config)
    welcome_message = f"""
    # Welcome to the MCP-powered Chat Demo

    This demo uses the Model Context Protocol (MCP) to standardize interactions with AI models.

    **MCP Server URL**: {mcp_url}

    ## Available models:
    {', '.join(available_models)}

    ## Current Settings:
    - **Model**: gpt-4o-mini
    - **Temperature**: 0.7
    - **Streaming**: Enabled

    ## Commands:
    - Type `/model <model_name>` to change the model (e.g., `/model gpt-4`)
    - Type `/temp <number>` to change temperature (e.g., `/temp 0.5`)
    - Type `/streaming on` or `/streaming off` to toggle streaming

    You can send messages and the application will route them through the MCP server.
    """

    # Send welcome message without actions (they're causing errors)
    welcome_msg = cl.Message(content=welcome_message)
    await welcome_msg.send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages using MCP."""
    global mcp_handler, available_models

    # Check for command messages
    msg_content = message.content.strip()

    # Handle model change command
    if msg_content.startswith("/model "):
        requested_model = msg_content[7:].strip()
        if requested_model in available_models:
            cl.user_session.set("model", requested_model)
            await cl.Message(content=f"Model changed to {requested_model}").send()
        else:
            models_list = ", ".join(available_models)
            await cl.Message(
                content=f"Model '{requested_model}' not available. Choose from: {models_list}"
            ).send()
        return

    # Handle temperature change command
    if msg_content.startswith("/temp "):
        try:
            temp = float(msg_content[6:].strip())
            if 0 <= temp <= 1:
                cl.user_session.set("temperature", temp)
                await cl.Message(content=f"Temperature set to {temp}").send()
            else:
                await cl.Message(content="Temperature must be between 0 and 1").send()
        except ValueError:
            await cl.Message(
                content="Invalid temperature value. Use a number between 0 and 1."
            ).send()
        return

    # Handle streaming toggle command
    if msg_content.startswith("/streaming "):
        option = msg_content[11:].strip().lower()
        if option in ["on", "true", "yes", "1"]:
            cl.user_session.set("streaming", True)
            await cl.Message(content="Streaming enabled").send()
        elif option in ["off", "false", "no", "0"]:
            cl.user_session.set("streaming", False)
            await cl.Message(content="Streaming disabled").send()
        else:
            await cl.Message(content="Invalid option. Use 'on' or 'off'.").send()
        return

    # Handle regular messages
    # Get message history
    message_history = cl.user_session.get("message_history", [])

    # Add user message to history
    message_history.append({"role": "user", "content": msg_content})

    # Get current settings
    model = cl.user_session.get("model", "gpt-4o-mini")
    temperature = cl.user_session.get("temperature", 0.7)
    streaming = cl.user_session.get("streaming", True)

    # Create empty message for response
    response_message = cl.Message(content="")
    await response_message.send()

    try:
        if streaming:
            # Use the MCP handler for streaming
            response_text = await mcp_handler.handle_message(
                message_history=message_history,
                current_message=response_message,
                model=model,
                temperature=temperature,
            )
        else:
            # For non-streaming, manually call MCP API
            # Display thinking indicator
            await response_message.update(content="Thinking...")

            # Call MCP API directly
            response_text = await call_mcp_api(
                messages=message_history,
                model=model,
                temperature=temperature,
                stream=False,
            )

            # Update message with full response
            await response_message.update(content=response_text)

        # Add response to message history
        message_history.append({"role": "assistant", "content": response_text})
        cl.user_session.set("message_history", message_history)

        # Update the message with the final content
        await response_message.update()

    except Exception as e:
        error_message = f"Error: {str(e)}"
        await response_message.update(content=error_message)


if __name__ == "__main__":
    # This allows running the demo directly
    import sys

    # Install dependencies if needed
    try:
        import tiktoken
    except ImportError:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "uv", "pip", "install", "tiktoken"]
        )

    # Run the Chainlit app directly
    import os

    os.system(f"chainlit run {__file__}")
