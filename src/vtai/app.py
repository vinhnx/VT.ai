import chainlit as cl
import config as conf
import litellm

from chainlit.input_widget import Select, Slider, Tags
from llm_profile_builder import build_llm_profile

litellm.model_alias_map = conf.MODEL_ALIAS_MAP


@cl.on_chat_start
async def start_chat():
    # build llm profile
    await build_llm_profile(conf.ICONS_PROVIDER_MAP)

    # settings configuration
    settings = await cl.ChatSettings(
        [
            Select(
                id=conf.SETTINGS_LLM_MODEL,
                label="Choose LLM Model",
                values=conf.MODELS,
                initial_index=0,
            )
        ]
    ).send()

    # set selected LLM model for current settion's model
    cl.user_session.set(
        conf.SETTINGS_LLM_MODEL,
        settings[conf.SETTINGS_LLM_MODEL],
    )

    # prompt
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant who tries their best to answer questions: ",
    }

    cl.user_session.set("message_history", [system_message])


@cl.on_message
async def on_message(message: cl.Message):
    messages = cl.user_session.get("message_history") or []
    if len(message.elements) > 0:
        # multi-modal: upload files to process
        for element in message.elements:
            with open(element.path, "r") as uploaded_file:
                content = uploaded_file.read()

            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
            confirm_message = cl.Message(content=f"Uploaded file: {element.name}")
            await confirm_message.send()

    model = str(cl.user_session.get(conf.SETTINGS_LLM_MODEL) or conf.DEFAULT_MODEL)
    msg = cl.Message(content="", author=model)
    await msg.send()

    messages.append(
        {
            "role": "user",
            "content": message.content,
        }
    )

    # trigger async litellm model with message
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        stream=True,
    )

    async for chunk in response:
        if chunk:
            content = chunk.choices[0].delta.content
            if content:
                await msg.stream_token(content)

    messages.append(
        {
            "role": "assistant",
            "content": msg.content,
        }
    )
    await msg.update()


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set(
        conf.SETTINGS_LLM_MODEL,
        settings[conf.SETTINGS_LLM_MODEL],
    )
