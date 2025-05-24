from typing import Any, Dict, List

import chainlit as cl

from vtai.utils import llm_providers_config as conf
from vtai.utils.error_handlers import safe_execution
from vtai.utils.llm_providers_config import (
    SETTINGS_CHAT_MODEL,
    SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING,
)
from vtai.utils.user_session_helper import get_setting, update_message_history_from_user


async def handle_conversation(
    message: cl.Message,
    messages: List[Dict[str, str]],
    route_layer: Any,
    user_keys: dict = None,
) -> None:
    """
    Handles text-based conversations with the user.
    Routes the conversation based on settings and semantic understanding.

    Args:
            message: The user message object
            messages: The conversation history
            route_layer: The semantic router layer
            user_keys: User-specific API keys for BYOK
    """
    model = get_setting(SETTINGS_CHAT_MODEL)
    # Use "assistant" as the author name to match the avatar file in /public/avatars/
    msg = cl.Message(content="")
    await msg.send()

    query = message.content
    update_message_history_from_user(query)

    async with safe_execution(operation_name="conversation handling"):
        use_dynamic_conversation_routing = get_setting(
            SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING
        )

        if use_dynamic_conversation_routing and route_layer:
            # This will still call into conversation_handlers for dynamic routing, but that's fine
            from vtai.utils.conversation_handlers import (
                handle_dynamic_conversation_routing,
            )

            await handle_dynamic_conversation_routing(
                messages,
                model,
                msg,
                query,
                route_layer,
                user_keys=user_keys,
            )
        else:
            await handle_trigger_async_chat(
                llm_model=model,
                messages=messages,
                current_message=msg,
                user_keys=user_keys,
            )


async def handle_trigger_async_chat(
    llm_model: str,
    messages: list[dict[str, str]],
    current_message: cl.Message,
    user_keys: dict = None,
) -> None:
    """
    Triggers an asynchronous chat completion using the specified LLM model.
    Streams the response back to the user and updates the message history.

    Args:
        llm_model: The LLM model to use
        messages: The conversation history messages
        current_message: The chainlit message object to stream response to
        user_keys: User-specific keys for BYOK
    """
    # Lazy import to avoid circular import
    from vtai.utils.conversation_handlers import (
        create_message_actions,
        update_message_history_from_assistant,
        use_chat_completion_api,
    )
    from vtai.utils.credits import get_user_credits_info
    from vtai.utils.user_session_helper import get_user_id

    user_id = get_user_id()
    if user_id:
        from vtai.utils.credits import check_user_can_chat

        if not check_user_can_chat(user_id):
            current_message.content = "\u26a0\ufe0f You have used all your free daily credits. Credits reset daily at midnight UTC. Upgrade for more!"
            await current_message.update()
            return

    temperature = get_setting(conf.SETTINGS_TEMPERATURE)
    top_p = get_setting(conf.SETTINGS_TOP_P)

    async def on_timeout():
        await current_message.stream_token(
            "\n\nI apologize, but the response timed out. Please try again with a shorter query."
        )
        await current_message.update()

    async with safe_execution(
        operation_name=f"chat completion with model {llm_model}", on_timeout=on_timeout
    ):
        await use_chat_completion_api(
            model=llm_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            stream_callback=current_message.stream_token,
            user_keys=user_keys,
        )

        content = current_message.content
        update_message_history_from_assistant(content)
        current_message.actions = create_message_actions(content, llm_model)

        # Add a cl.Action with current credit info after each successful response
        user_id = get_user_id()
        if user_id:
            credits_info = get_user_credits_info(user_id)
            credit_label = (
                f"Credits: {credits_info['credits_left']}/{credits_info['max_credits']}"
            )
            current_message.actions.append(
                cl.Action(
                    icon="lucide:coins",
                    name="user_credits_info",
                    label=credit_label,
                    payload={"value": credit_label},
                    description=f"Daily usage credits. Reset: {credits_info['reset_time'][:16].replace('T', ' ')} UTC",
                )
            )
        await current_message.update()
