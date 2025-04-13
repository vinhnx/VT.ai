"""
Assistant tools handling utilities for VT application.

Processes OpenAI Assistant tool calls and manages thread messages.
"""

from datetime import datetime
from typing import Any, Dict, Optional

import chainlit as cl
from openai.types.beta.threads import ImageFileContentBlock, Message, TextContentBlock
from openai.types.beta.threads.runs import RunStep

from vtai.utils import constants as const
from vtai.utils.config import logger
from vtai.utils.user_session_helper import get_setting


# Action callback for changing the model
@cl.action_callback("change_model_action")
async def on_change_model(action: cl.Action):
    """
    Handle the change model action click.
    Shows an inline text notification instructing the user how to change the model.

    Args:
        action: The action that was triggered
    """
    await action.remove()

    text_content = "To change the model, please click on the settings gear icon in the input bar at the bottom of the screen and select a different model from the 'Chat Model' dropdown."
    elements = [
        cl.Text(name="Change Language Model", content=text_content, display="inline")
    ]

    await cl.Message(content="", author=const.APP_NAME, elements=elements).send()


async def process_thread_message(
    message_references: Dict[str, cl.Message],
    thread_message: Message,
    async_openai_client: Any,
) -> None:
    """
    Process a thread message from the OpenAI Assistant API.

    Args:
        message_references: Dictionary of message references
        thread_message: The thread message to process
        async_openai_client: AsyncOpenAI client instance
    """
    for idx, content_message in enumerate(thread_message.content):
        id = thread_message.id + str(idx)

        if isinstance(content_message, TextContentBlock):
            if id in message_references:
                msg = message_references[id]
                msg.content = content_message.text.value
                await msg.update()
            else:
                message_references[id] = cl.Message(
                    author=const.APP_NAME,
                    content=content_message.text.value,
                )

                res_message = message_references[id].content

                # Get current model name for the action label
                current_model = get_setting("settings_chat_model") or "Chat Model"

                # Create actions list
                actions = []

                # Add TTS action if enabled
                enable_tts_response = get_setting("settings_enable_tts_response")
                if enable_tts_response:
                    actions.append(
                        cl.Action(
                            name="speak_chat_response_action",
                            payload={"value": res_message},
                            label="Speak response",
                        )
                    )

                # Add model change action
                actions.append(
                    cl.Action(
                        name="change_model_action",
                        payload={"value": res_message},
                        label=f"Using: {current_model}",
                        description="Click to change model in settings",
                    )
                )

                # Set the actions on the message
                message_references[id].actions = actions

                await message_references[id].send()

        elif isinstance(content_message, ImageFileContentBlock):
            image_id = content_message.image_file.file_id
            try:
                response = (
                    await async_openai_client.files.with_raw_response.retrieve_content(
                        image_id
                    )
                )
                elements = [
                    cl.Image(
                        content=response.content,
                        display="inline",
                        size="large",
                    ),
                ]

                if id not in message_references:
                    message_references[id] = cl.Message(
                        author=const.APP_NAME,
                        content="",
                        elements=elements,
                    )

                    res_message = message_references[id].content

                    # Get current model name for the action label
                    current_model = get_setting("settings_chat_model") or "Chat Model"

                    # Create actions list
                    actions = []

                    # Add TTS action if enabled
                    enable_tts_response = get_setting("settings_enable_tts_response")
                    if enable_tts_response:
                        actions.append(
                            cl.Action(
                                name="speak_chat_response_action",
                                payload={"value": res_message},
                                label="Speak response",
                            )
                        )

                    # Add model change action
                    actions.append(
                        cl.Action(
                            name="change_model_action",
                            payload={"value": res_message},
                            label=f"Using: {current_model}",
                            description="Click to change model in settings",
                        )
                    )

                    # Set the actions on the message
                    message_references[id].actions = actions

                    await message_references[id].send()
            except Exception as e:
                logger.error(f"Error retrieving image content: {e}")
                await cl.Message(content=f"Failed to load image: {str(e)}").send()
        else:
            logger.warning(f"Unknown message type: {type(content_message)}")


async def process_tool_call(
    step_references: Dict[str, cl.Step],
    step: RunStep,
    tool_call: Any,
    name: str,
    input: Any,
    output: Any,
    show_input: Optional[str] = None,
) -> None:
    """
    Process a tool call from the OpenAI Assistant API.

    Args:
        step_references: Dictionary of step references
        step: The run step
        tool_call: The tool call to process
        name: Name of the tool
        input: Input to the tool
        output: Output from the tool
        show_input: Language for input formatting
    """
    cl_step = None
    update = False

    # Safely handle tool_call as both object and dict
    tool_call_id = getattr(tool_call, "id", None)
    if tool_call_id is None and isinstance(tool_call, dict):
        tool_call_id = tool_call.get("id")

    if tool_call_id not in step_references:
        cl_step = cl.Step(
            name=name,
            type="tool",
            parent_id=cl.context.current_step.id if cl.context.current_step else None,
            language=show_input,
        )
        step_references[tool_call_id] = cl_step
    else:
        update = True
        cl_step = step_references[tool_call_id]

    if step.created_at:
        cl_step.start = datetime.fromtimestamp(step.created_at).isoformat()
    if step.completed_at:
        cl_step.end = datetime.fromtimestamp(step.completed_at).isoformat()

    cl_step.input = input
    cl_step.output = output

    if update:
        await cl_step.update()
    else:
        await cl_step.send()
        await cl_step.send()
