"""
Creates and manages assistants for VT.ai.

This module provides functions to create, update, and manage OpenAI Assistants.
"""

import os
from typing import List, Optional

from openai import AsyncOpenAI
from openai.types.beta import Assistant

from vtai.assistants.tools import ASSISTANT_TOOLS
from vtai.utils import constants as const


async def create_assistant(
    client: AsyncOpenAI,
    name: str = const.APP_NAME,
    instructions: str = "You are a helpful, expert assistant with diverse capabilities.",
    tools: Optional[List[dict]] = None,
    model: str = "gpt-4o",
) -> Assistant:
    """
    Create or update an assistant with the specified configuration.

    Args:
        client: OpenAI async client
        name: Name of the assistant
        instructions: System instructions for the assistant
        tools: List of tool configurations (defaults to ASSISTANT_TOOLS)
        model: Model to use for the assistant

    Returns:
        Created or updated Assistant object
    """
    # Use the default tools list if none provided
    if tools is None:
        tools = ASSISTANT_TOOLS

    # Create assistant
    assistant = await client.beta.assistants.create(
        name=name,
        instructions=instructions,
        tools=tools,
        model=model,
    )

    return assistant


async def get_or_create_assistant(
    client: AsyncOpenAI,
    assistant_id: Optional[str] = None,
    name: str = const.APP_NAME,
    instructions: str = "You are a helpful, expert assistant with diverse capabilities.",
    tools: Optional[List[dict]] = None,
    model: str = "gpt-4o",
) -> Assistant:
    """
    Get an existing assistant by ID or create a new one if not found.

    Args:
        client: OpenAI async client
        assistant_id: Optional existing assistant ID
        name: Name for the assistant if creating new
        instructions: System instructions for the assistant
        tools: List of tool configurations (defaults to ASSISTANT_TOOLS)
        model: Model to use for the assistant

    Returns:
        Retrieved or created Assistant object
    """
    # Try to retrieve existing assistant if ID is provided
    if assistant_id:
        try:
            return await client.beta.assistants.retrieve(assistant_id)
        except Exception as e:
            # If retrieval fails, log the error and proceed to create a new assistant
            print(
                f"Error retrieving assistant {assistant_id}: {e}. Creating new assistant."
            )

    # Create a new assistant
    return await create_assistant(
        client=client, name=name, instructions=instructions, tools=tools, model=model
    )
