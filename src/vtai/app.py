import os
from getpass import getpass

import chainlit as cl
import config as conf
import litellm
from chainlit.input_widget import Select
from llm_profile_builder import build_llm_profile
from semantic_router.layer import RouteLayer
from constants import SemanticRouterType

# check for OpenAI API key, default default we will use GPT-3.5-turbo model
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
    "Enter OpenAI API Key: "
)

# map litellm model aliases
litellm.model_alias_map = conf.MODEL_ALIAS_MAP


# semanticrouter - Semantic Router is a superfast decision-making layer for LLMs and agents
route_layer = RouteLayer.from_json("semantic_route_layers.json")


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
    query = message.content

    messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    route_choice = route_layer(query)
    route_choice_name = route_choice.name

    # detemine conversation routing
    if route_choice_name == SemanticRouterType.IMAGE_GEN:
        image_gen_model = "dall-e-3"
        image_response = litellm.image_generation(
            prompt=query,
            model=image_gen_model,
        )

        image_gen_data = image_response["data"][0]
        image_url = image_gen_data["url"]
        revised_prompt = image_gen_data["revised_prompt"]

        image = cl.Image(
            url=image_url,
            name=query,
            display="inline",
        )
        revised_prompt_text = cl.Text(
            name="Description", content=revised_prompt, display="inline"
        )

        msg = cl.Message(
            author=image_gen_model,
            content="Sure, here it is!",
            elements=[
                image,
                revised_prompt_text,
            ],
        )
        await msg.send()

        messages.append(
            {
                "role": "assistant",
                "content": revised_prompt,
            }
        )

    else:
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
