import os
from getpass import getpass

import chainlit as cl
import litellm
import llms_config as conf
from chainlit.input_widget import Select, Switch
from constants import SemanticRouterType
from litellm.utils import trim_messages
from llm_profile_builder import build_llm_profile
from semantic_router.layer import RouteLayer
from url_extractor import extract_url

import dotenv

dotenv.load_dotenv()


# check for OpenAI API key, default default we will use GPT-3.5-turbo model
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
    "Enter OpenAI API Key: "
)

# ---
# MY TODO LIST
# TODO: https://github.com/Chainlit/cookbook/tree/main/llava
# TODO: support Vision LLAVA, GPT, GEMINI
# TODO: support Audio transcript: WHISPER
# TODO: token count, model name https://docs.litellm.ai/docs/completion/output
# TODO: action: https://docs.chainlit.io/api-reference/ask/ask-for-action
# TODO: chat response Whisper like ChatGPT.!
# TODO: footer "vt.ai can make mistakes. Consider checking important information."
# TODO: TaskList https://docs.chainlit.io/api-reference/elements/tasklist
# TODO: https://docs.chainlit.io/api-reference/data-persistence/custom-data-layer
# TODO: "Chat Profiles are useful if you want to let your users choose from a list of predefined configured assistants. For example, you can define a chat profile for a support chat, a sales chat, or a chat for a specific product." https://docs.chainlit.io/advanced-features/chat-profiles
# TODO: callback https://docs.chainlit.io/concepts/action#define-a-python-callback
# TODO: customize https://docs.chainlit.io/customisation/overview
# TODO: config https://docs.chainlit.io/backend/config/overview
# TODO: sync/async https://docs.chainlit.io/guides/sync-async
# TODO: function call https://docs.litellm.ai/docs/completion/function_call
# TODO https://docs.litellm.ai/docs/completion/reliable_completions
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
    settings = await __build_settings()

    # set selected LLM model for current settion's model
    __config_chat_session(settings)


@cl.on_message
async def on_message(message: cl.Message):
    # retrieve message memory
    messages = cl.user_session.get("message_history") or []

    if len(message.elements) > 0:
        await __handle_files_upload(message, messages)
    else:
        await __handle_conversation(message, messages)


async def __handle_conversation(message, messages):
    model = __get_settings(conf.SETTINGS_CHAT_MODEL)
    msg = cl.Message(content="", author=model)
    await msg.send()

    query = message.content
    messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    use_dynamic_conversation_routing = __get_settings(
        conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING
    )

    if use_dynamic_conversation_routing is True:
        await __handle_dynamic_conversation_routing(messages, model, msg, query)

    else:
        await __handle_trigger_async_chat(
            llm_model=model,
            messages=messages,
            current_message=msg,
        )


@cl.on_settings_update
async def update_settings(settings):
    cl.user_session.set(conf.SETTINGS_CHAT_MODEL, settings[conf.SETTINGS_CHAT_MODEL])
    cl.user_session.set(
        conf.SETTINGS_VISION_MODEL, settings[conf.SETTINGS_VISION_MODEL]
    )
    cl.user_session.set(
        conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING,
        settings[conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING],
    )


async def __handle_trigger_async_chat(llm_model, messages, current_message):
    # trigger async litellm model with message
    try:
        stream = await litellm.acompletion(
            model=llm_model,
            messages=messages,
            stream=True,
            num_retries=2,
        )

    except Exception as e:
        await __handle_exception_error(llm_model, messages, current_message, e)
        return

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await current_message.stream_token(token)

    content = current_message.content
    messages.append(
        {
            "role": "assistant",
            "content": content,
        }
    )

    await current_message.update()


async def __handle_exception_error(e):
    print(f"Error type: {type(e)}, Error: {e}")
    await cl.Message(
        content=f"Something went wrong. Error type: {type(e)}, Error: {e}",
    ).send()


def __config_chat_session(settings):
    cl.user_session.set(
        conf.SETTINGS_CHAT_MODEL,
        settings[conf.SETTINGS_CHAT_MODEL],
    )

    # prompt
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant who tries their best to answer questions: ",
    }

    cl.user_session.set("message_history", [system_message])


async def __build_settings():
    settings = await cl.ChatSettings(
        [
            Select(
                id=conf.SETTINGS_CHAT_MODEL,
                label="Chat Model",
                values=conf.MODELS,
                initial_value=conf.DEFAULT_MODEL,
            ),
            Select(
                id=conf.SETTINGS_VISION_MODEL,
                label="Vision Model",
                values=conf.VISION_MODEL_MODELS,
                initial_value=conf.DEFAULT_VISION_MODEL,
            ),
            Switch(
                id=conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING,
                label="Use dynamic conversation routing",
                description=f"[Beta] You can turn on this option to enable dynamic conversation routing. For example, when in a middle of the chat when you ask something like `Help me generate a cute dog image`, the app will automatically use Image Generation LLM Model selection to generate a image for you, using OpenAI DALL·E 3. Note: this action requires OpenAI API key. Default is {conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE}",
                initial=conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE,
            ),
            Switch(
                id=conf.SETTINGS_TRIMMED_MESSAGES,
                label="Trimming Input Messages",
                description="Ensure messages does not exceed a model's token limit",
                initial=conf.SETTINGS_TRIMMED_MESSAGES_DEFAULT_VALUE,
            ),
        ]
    ).send()

    return settings


async def __handle_trigger_async_image_gen(messages, query):
    image_gen_model = conf.DEFAULT_IMAGE_GEN_MODEL
    message = cl.Message(
        content=f"Sure, I will use `{image_gen_model}` model to generate the image for you. Please wait a moment..",
        author=image_gen_model,
    )
    await message.send()

    try:
        image_response = await litellm.aimage_generation(
            prompt=query,
            model=image_gen_model,
        )

    except Exception as e:
        await __handle_exception_error(image_gen_model, messages, message, e)
        return

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

    await message.send()


async def __handle_files_upload(message, messages):
    if not message.elements:
        await cl.Message(content="No file attached").send()
        return

    prompt = message.content

    # Processing files
    for file in message.elements:
        if "image" in file.mime:
            # Read the first image

            path = file.path
            with open(path, "r") as f:
                pass

            await __handle_vision(path, prompt=prompt, messages=messages, is_local=True)
            break

        elif "text" in file.mime:
            with open(file.path, "r") as uploaded_file:
                content = uploaded_file.read()

            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )

            await __handle_conversation(message, messages)
            break

        else:
            await cl.Message(
                content=f"The file `{file.name}` with type: `{file.mime}` is currently not supported. Please try another file.",
            ).send()


async def __handle_dynamic_conversation_routing(messages, model, msg, query):
    route_choice = route_layer(query)
    route_choice_name = route_choice.name

    should_trimmed_messages = __get_settings(conf.SETTINGS_TRIMMED_MESSAGES)
    if should_trimmed_messages:
        messages = trim_messages(messages, model)

    # determine conversation routing
    print(f"""💡
          Query: {query}
          Is classified as route: {route_choice_name}
          running router...""")

    if route_choice_name == SemanticRouterType.IMAGE_GENERATION:
        print(f"""💡
            Running route_choice_name: {route_choice_name}.
            Processing image generation...""")
        await __handle_trigger_async_image_gen(messages, query)

    elif route_choice_name == SemanticRouterType.VISION_IMAGE_PROCESSING:
        urls = extract_url(query)
        if len(urls) > 0:
            print(f"""💡
                Running route_choice_name: {route_choice_name}.
                Received image urls/paths.
                Processing with Vision model...""")

            url = urls[0]
            await __handle_vision(
                input_image=url, prompt=query, messages=messages, is_local=False
            )
        else:
            print(f"""💡
                Running route_choice_name: {route_choice_name}.
                Received no image urls/paths.
                Processing with async chat...""")

            await __handle_trigger_async_chat(
                llm_model=model,
                messages=messages,
                current_message=msg,
            )

    else:
        print(f"""💡
            Running route_choice_name: {route_choice_name}.
            Processing with async chat...""")
        await __handle_trigger_async_chat(
            llm_model=model,
            messages=messages,
            current_message=msg,
        )


async def __handle_vision(
    input_image,
    prompt,
    messages,
    is_local=False,
):
    vision_model = (
        conf.DEFAULT_VISION_MODEL
        if is_local
        else __get_settings(conf.SETTINGS_VISION_MODEL)
    )

    supports_vision = litellm.supports_vision(model=vision_model)

    if supports_vision is False:
        print(f"Unsupported vision model: {vision_model}")
        await cl.Message(
            content=f"The model `{vision_model}`, doesn't support Vision capability. You choose a different model in Settings.",
        ).send()
        return

    message = cl.Message(
        content=f"Sure, I will use `{vision_model}` model to identify the image for you. Please wait a moment..",
        author=vision_model,
    )

    await message.send()
    vresponse = await litellm.acompletion(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": input_image}},
                ],
            }
        ],
    )

    description = vresponse.choices[0].message.content
    messages.append(
        {
            "role": "assistant",
            "content": description,
        }
    )
    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    if is_local:
        image = cl.Image(
            path=input_image,
            name=prompt,
            display="inline",
        )
    else:
        image = cl.Image(
            url=input_image,
            name=prompt,
            display="inline",
        )

    revised_prompt_text = cl.Text(name="Explain", content=description, display="inline")
    message = cl.Message(
        author=vision_model,
        content="",
        elements=[
            image,
            revised_prompt_text,
        ],
    )

    await message.send()


def __get_settings(key):
    return cl.user_session.get("chat_settings")[key]
