"""
Utility function for handling DeepSeek Reasoner conversations.
This is meant to replace the existing handle_deepseek_reasoner_conversation function in conversation_handlers.py.
"""

import asyncio
import os
import time
from typing import Any, Dict, List

import chainlit as cl
from openai import AsyncOpenAI
from utils.config import logger

# Import the create_message_actions function which is needed by this handler
from utils.conversation_handlers import create_message_actions
from utils.error_handlers import safe_execution
from utils.usage_logger import get_user_id_from_session, log_usage_to_supabase
from utils.user_session_helper import (
    get_user_session_id,
    update_message_history_from_assistant,
    update_message_history_from_user,
)


async def handle_deepseek_reasoner_with_logging(
    message: cl.Message, messages: List[Dict[str, str]], route_layer: Any
) -> None:
    """
    Handles conversations using DeepSeek Reasoner model with its native reasoning capability.
    Shows the AI's reasoning process before presenting the final answer.
    Uses DeepSeek's reasoning_content attribute for the thinking process.

    Args:
        message: The user message object
        messages: The conversation history
        route_layer: The semantic router layer
    """
    # Track start time for the thinking duration
    start = time.time()

    # Get settings
    deepseek_api_key = os.getenv("DEEP_SEEK_API_KEY")
    if not deepseek_api_key:
        await cl.Message(
            content="DeepSeek API key not found. Please set the DEEP_SEEK_API_KEY environment variable."
        ).send()
        return

    model = "deepseek-reasoner"  # DeepSeek Reasoner model
    query = message.content.strip()
    update_message_history_from_user(query)

    # Create DeepSeek client
    client = AsyncOpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    # Variables that will be initialized in the context
    thinking_step = None
    final_answer = None
    thinking_completed = False

    # Variables to collect token usage estimates
    collected_reasoning = ""
    collected_content = ""

    async with safe_execution(
        operation_name=f"deepseek reasoner conversation with model {model}"
    ):
        # Prepare messages for the API call
        deepseek_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
        ]

        # Add conversation history
        for msg in messages:
            if msg["role"] != "system":  # Skip system messages from history
                deepseek_messages.append(msg)

        # Add the current user query
        deepseek_messages.append({"role": "user", "content": query})

        # Create the stream
        stream = await client.chat.completions.create(
            model=model,
            messages=deepseek_messages,
            stream=True,
        )

        # Start with a thinking step
        async with cl.Step(name="Thinking") as step:
            thinking_step = step
            # Create a message for the final answer, but don't send it yet
            final_answer = cl.Message(content="")

            # Process the reasoning content first
            async for chunk in stream:
                delta = chunk.choices[0].delta
                reasoning_content = getattr(delta, "reasoning_content", None)

                if reasoning_content is not None and not thinking_completed:
                    # Stream the reasoning content to the thinking step
                    await thinking_step.stream_token(reasoning_content)
                    collected_reasoning += reasoning_content
                elif not thinking_completed:
                    # Exit the thinking step when reasoning is complete
                    thought_for = round(time.time() - start)
                    thinking_step.name = f"Thought for {thought_for}s"
                    await thinking_step.update()
                    thinking_completed = True
                    break

        # Stream the final answer content
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                collected_content += delta.content
                await final_answer.stream_token(delta.content)

        # Calculate token usage (roughly)
        # For DeepSeek, we don't get the token counts directly, so we need to estimate
        total_input_chars = sum(len(m.get("content", "")) for m in deepseek_messages)
        total_output_chars = len(collected_content) + len(collected_reasoning)

        # Rough approximation: 1 token ≈ 4 characters for English text
        input_tokens_estimate = total_input_chars // 4
        output_tokens_estimate = total_output_chars // 4
        total_tokens_estimate = input_tokens_estimate + output_tokens_estimate

        # Log estimated token usage
        logger.info(
            f"Estimated token usage for DeepSeek Reasoner: input={input_tokens_estimate}, "
            f"output={output_tokens_estimate}, total={total_tokens_estimate}"
        )

        # Log usage to Supabase if not using password auth
        try:
            session_id = get_user_session_id()
            user_id = get_user_id_from_session()

            await log_usage_to_supabase(
                user_id=user_id,
                session_id=session_id,
                model_name=model,
                input_tokens=input_tokens_estimate,
                output_tokens=output_tokens_estimate,
                total_tokens=total_tokens_estimate,
                cost=None,  # Cost calculation to be implemented later
            )
        except Exception as e:
            logger.error(f"Error logging DeepSeek usage to Supabase: {e}")

        # Send the final answer after thinking is complete
        if not final_answer.content:
            # If no final answer was provided, create a fallback message
            await cl.Message(
                content="I've thought about this but don't have a specific answer to provide.",
            ).send()
        else:
            # Update message history and add TTS action
            content = final_answer.content
            update_message_history_from_assistant(content)

            # Set the actions on the message using the helper function
            final_answer.actions = create_message_actions(content, model)

            await final_answer.send()
            await final_answer.send()
