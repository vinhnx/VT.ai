import os
from getpass import getpass

import chainlit as cl
import config as conf
import litellm
from chainlit.input_widget import Select, Switch
from constants import SemanticRouterType
from llm_profile_builder import build_llm_profile
from semantic_router.layer import RouteLayer

# check for OpenAI API key, default default we will use GPT-3.5-turbo model
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
    "Enter OpenAI API Key: "
)

# ---
# MY TODO LIST
# TODO: multi-modal drag and drop https://docs.chainlit.io/advanced-features/multi-modal
# TODO: https://github.com/Chainlit/cookbook/tree/main/llava
# TODO: Disabling Multi-Modal Functionality. If you wish to disable this feature (which would prevent users from attaching files to their messages), you can do so by setting features.multi_modal=false in your Chainlit config file.
# TODO: support Vision LLAVA, GPT, GEMINI
# TODO: support Audio transcript: WHISPER
# TODO: token count, model name https://docs.litellm.ai/docs/completion/output
# TODO: error handling https://docs.litellm.ai/docs/exception_mapping
# TODO: retry/regenerate response
# TODO: action: https://docs.chainlit.io/api-reference/ask/ask-for-action
# TODO: chat response Whisper like ChatGPT.!
# TODO: footer "vt.ai can make mistakes. Consider checking important information."
# TODO: TaskList https://docs.chainlit.io/api-reference/elements/tasklist
# TODO: https://docs.chainlit.io/api-reference/data-persistence/custom-data-layer
# TODO: "Chat Profiles are useful if you want to let your users choose from a list of predefined configured assistants. For example, you can define a chat profile for a support chat, a sales chat, or a chat for a specific product." https://docs.chainlit.io/advanced-features/chat-profiles
# TODO: toast https://docs.chainlit.io/concepts/action#toaster
# TODO: callback https://docs.chainlit.io/concepts/action#define-a-python-callback
# TODO: customize https://docs.chainlit.io/customisation/overview
# TODO: config https://docs.chainlit.io/backend/config/overview
# TODO: sync/async https://docs.chainlit.io/guides/sync-async

# ---
# Advanced
# TODO: Auth https://docs.chainlit.io/authentication/overview
# TODO: Data persistence https://docs.chainlit.io/data-persistence/overview
# TODO: custom endpoint https://docs.chainlit.io/backend/custom-endpoint
# TODO: deploy https://docs.chainlit.io/deployment/tutorials
# TODO: copilot chat widget https://docs.chainlit.io/deployment/copilot

# map litellm model aliases
litellm.model_alias_map = conf.MODEL_ALIAS_MAP


# semanticrouter - Semantic Router is a superfast decision-making layer for LLMs and agents
route_layer = RouteLayer.from_json("semantic_route_layers.json")


@cl.on_chat_start
async def start_chat():
    # build llm profile
    await build_llm_profile(conf.ICONS_PROVIDER_MAP)

    # settings configuration
    settings = await build_settings()

    # set selected LLM model for current settion's model
    config_chat_session(settings)


@cl.on_message
async def on_message(message: cl.Message):
    # retrive message memory
    messages = cl.user_session.get("message_history") or []

    if len(message.elements) > 0:
        await handle_files_upload(message, messages)

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

    use_dynamic_conversation_routing = cl.user_session.get(
        conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING
    )

    if use_dynamic_conversation_routing:
        await handle_dynamic_conversation_routing(messages, model, msg, query)

    else:
        await handle_trigger_async_chat(
            llm_model=model,
            messages=messages,
            current_message=msg,
        )


@cl.on_settings_update
async def setup_agent(settings):
    print(f"setup_agent: {settings}")
    cl.user_session.set(conf.SETTINGS_LLM_MODEL, settings[conf.SETTINGS_LLM_MODEL])

    cl.user_session.set(
        conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING,
        settings[conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING],
    )


async def handle_trigger_async_chat(llm_model, messages, current_message):
    # trigger async litellm model with message
    stream = await litellm.acompletion(
        model=llm_model,
        messages=messages,
        stream=True,
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await current_message.stream_token(token)

    messages.append(
        {
            "role": "assistant",
            "content": current_message.content,
        }
    )
    await current_message.update()


def config_chat_session(settings):
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


async def build_settings():
    settings = await cl.ChatSettings(
        [
            Select(
                id=conf.SETTINGS_LLM_MODEL,
                label="Chat LLM Model",
                values=conf.MODELS,
                initial_value=conf.DEFAULT_MODEL,
            ),
            Switch(
                id=conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING,
                label="Use dynamic conversation routing",
                description=f"[Beta] You can turn on this option to enable dynamic conversation routing. For example, when in a middle of the chat when you ask something like `Help me generate a cute dog image`, the app will automatically use Image Generation LLM Model selection to generate a image for you, using OpenAI DALL·E 3. Note: this action requires OpenAI API key. Default is {conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE}",
                initial=conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE,
            ),
        ]
    ).send()

    return settings


async def handle_trigger_async_image_gen(messages, query):
    image_gen_model = conf.DEFAULT_IMAGE_GEN_MODEL
    message = cl.Message(
        content=f"Sure, I will use `{image_gen_model}` model to generate the image for you. Please wait a moment..",
        author=image_gen_model,
    )
    await message.send()

    image_response = await litellm.aimage_generation(
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

    message = cl.Message(
        author=image_gen_model,
        content="The image is completed, here it is!",
        elements=[
            image,
            revised_prompt_text,
        ],
    )

    messages.append(
        {
            "role": "assistant",
            "content": revised_prompt,
        }
    )

    await message.send()


async def handle_files_upload(message, messages):
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


async def handle_dynamic_conversation_routing(messages, model, msg, query):
    route_choice = route_layer(query)
    route_choice_name = route_choice.name

    # detemine conversation routing
    if route_choice_name == SemanticRouterType.IMAGE_GEN:
        await handle_trigger_async_image_gen(messages, query)
    else:
        await handle_trigger_async_chat(
            llm_model=model,
            messages=messages,
            current_message=msg,
        )
