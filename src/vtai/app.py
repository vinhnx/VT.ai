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
            ),
            Slider(
                id=conf.SETTINGS_LLM_PARAMS_TEMPERATURE,
                label="temperature",
                description="""predictable and typical responses, while higher values encourage more diverse and less common responses. At 0, the model always gives the same response for a given input.
                
                Optional, float, 0.0 to 2.0. Default: 1.0
                """,
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id=conf.SETTINGS_LLM_PARAMS_TOP_P,
                label="top_p",
                description="""This setting limits the model's choices to a percentage of likely tokens: only the top tokens whose probabilities add up to P. A lower value makes the model's responses more predictable, while the default setting allows for a full range of token choices. Think of it like a dynamic Top-K.
                
                Optional, float, 0.0 to 1.0. Default: 1.0
                """,
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id=conf.SETTINGS_LLM_PARAMS_MAX_TOKENS,
                label="max_tokens",
                description="""This sets the upper limit for the number of tokens the model can generate in response. It won't produce more than this limit. The maximum value is the context length minus the prompt length.
                
                Optional, integer, 1 or above
                """,
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),
            Tags(
                id=conf.SETTINGS_LLM_PARAMS_STOP_SEQUENCE,
                label="stop",
                description="""Stop generation immediately if the model encounter any token specified in the stop array. Optional, array
                """,
            ),
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
    temperature = cl.user_session.get(conf.SETTINGS_LLM_PARAMS_TEMPERATURE)
    top_p = cl.user_session.get(conf.SETTINGS_LLM_PARAMS_TOP_P)
    max_tokens = cl.user_session.get(conf.SETTINGS_LLM_PARAMS_MAX_TOKENS)
    stop_sequence = cl.user_session.get(conf.SETTINGS_LLM_PARAMS_STOP_SEQUENCE)
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop_sequence,
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

    cl.user_session.set(
        conf.SETTINGS_LLM_PARAMS_TEMPERATURE,
        settings[conf.SETTINGS_LLM_PARAMS_TEMPERATURE],
    )

    cl.user_session.set(
        conf.SETTINGS_LLM_PARAMS_TOP_P,
        settings[conf.SETTINGS_LLM_PARAMS_TOP_P],
    )

    cl.user_session.set(
        conf.SETTINGS_LLM_PARAMS_MAX_TOKENS,
        settings[conf.SETTINGS_LLM_PARAMS_MAX_TOKENS],
    )

    cl.user_session.set(
        conf.SETTINGS_LLM_PARAMS_STOP_SEQUENCE,
        settings[conf.SETTINGS_LLM_PARAMS_STOP_SEQUENCE],
    )
