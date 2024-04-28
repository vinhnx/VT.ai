import os
import pathlib
import tempfile
from getpass import getpass
from typing import Any, Dict, List

import chainlit as cl
import dotenv
import litellm
import utils.constants as const
from assistants.mino.mino import MinoAssistant
from assistants.mino.mino import INSTRUCTIONS
from litellm.utils import trim_messages
from openai import AsyncOpenAI, OpenAI
from router.constants import SemanticRouterType
from semantic_router.layer import RouteLayer
from utils import llm_config as conf
from utils.chat_profile import AppChatProfileType
from utils.llm_profile_builder import build_llm_profile
from utils.settings_builder import build_settings
from utils.url_extractor import extract_url

# Load .env
dotenv.load_dotenv(dotenv.find_dotenv())

# Model alias map for litellm
litellm.model_alias_map = conf.MODEL_ALIAS_MAP

# Load semantic router layer from JSON file
route_layer = RouteLayer.from_json("./src/router/layers.json")

# Create temporary directory for TTS audio files
temp_dir = tempfile.TemporaryDirectory()

# Set LLM Providers API Keys from environment variable or user input
# OpenAI - API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
    "Enter OpenAI API Key: "
)
# OpenAI - Organization ID - track expense
os.environ["OPENAI_ORGANIZATION"] = os.getenv("OPENAI_ORGANIZATION") or getpass(
    "Enter OpenAI Orginazation ID, for billing management. You can skip this."
)  # OPTIONAL

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY") or getpass(
    "Enter Google Gemini API Key, this is used for Vision capability. You can skip this: "
)


# Initialize OpenAI client
openai_client = OpenAI(max_retries=2)
async_openai_client = AsyncOpenAI(max_retries=2)


# constants
APP_NAME = const.APP_NAME

# NOTE: 💡 Check ./TODO file for TODO list


@cl.set_chat_profiles
async def build_chat_profile():
    return conf.CHAT_PROFILES


@cl.on_chat_start
async def start_chat():
    """
    Initializes the chat session.
    Builds LLM profiles, configures chat settings, and sets initial system message.
    """
    # build llm profile
    await build_llm_profile(conf.ICONS_PROVIDER_MAP)

    # settings configuration
    settings = await build_settings()

    # set selected LLM model for current settion's model
    await __config_chat_session__(settings)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Handles incoming messages from the user.
    Processes text messages, file attachment, and routes conversations accordingly.
    """

    # Chatbot memory
    messages = cl.user_session.get("message_history") or []  # Get message history

    if len(message.elements) > 0:
        await __handle_files_attachment__(message, messages)  # Process file attachments
    else:
        await __handle_conversation(message, messages)  # Process text messages


@cl.on_settings_update
async def update_settings(settings: Dict[str, Any]) -> None:
    """
    Updates chat settings based on user preferences.
    """

    if settings_temperature := settings[conf.SETTINGS_TEMPERATURE]:
        cl.user_session.set(conf.SETTINGS_TEMPERATURE, settings_temperature)

    if settings_top_p := settings[conf.SETTINGS_TOP_P]:
        cl.user_session.set(conf.SETTINGS_TOP_P, settings_top_p)

    cl.user_session.set(conf.SETTINGS_CHAT_MODEL, settings[conf.SETTINGS_CHAT_MODEL])
    cl.user_session.set(
        conf.SETTINGS_IMAGE_GEN_IMAGE_STYLE,
        settings[conf.SETTINGS_IMAGE_GEN_IMAGE_STYLE],
    )
    cl.user_session.set(
        conf.SETTINGS_IMAGE_GEN_IMAGE_QUALITY,
        settings[conf.SETTINGS_IMAGE_GEN_IMAGE_QUALITY],
    )
    cl.user_session.set(
        conf.SETTINGS_VISION_MODEL, settings[conf.SETTINGS_VISION_MODEL]
    )
    cl.user_session.set(
        conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING,
        settings[conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING],
    )
    cl.user_session.set(conf.SETTINGS_TTS_MODEL, settings[conf.SETTINGS_TTS_MODEL])
    cl.user_session.set(
        conf.SETTINGS_TTS_VOICE_PRESET_MODEL,
        settings[conf.SETTINGS_TTS_VOICE_PRESET_MODEL],
    )
    cl.user_session.set(
        conf.SETTINGS_ENABLE_TTS_RESPONSE, settings[conf.SETTINGS_ENABLE_TTS_RESPONSE]
    )


@cl.action_callback("speak_chat_response_action")
async def on_speak_chat_response(action: cl.Action) -> None:
    """
    Handles the action triggered by the user.
    """
    await action.remove()
    value = action.value
    return await __handle_tts_response__(value)


async def __handle_tts_response__(context: str) -> None:
    """
    Generates and sends a TTS audio response using OpenAI's Audio API.
    """
    enable_tts_response = __get_settings(conf.SETTINGS_ENABLE_TTS_RESPONSE)
    if enable_tts_response is False:
        return

    if len(context) == 0:
        return

    model = __get_settings(conf.SETTINGS_TTS_MODEL)
    voice = __get_settings(conf.SETTINGS_TTS_VOICE_PRESET_MODEL)

    with openai_client.audio.speech.with_streaming_response.create(
        model=model, voice=voice, input=context
    ) as response:
        temp_filepath = os.path.join(temp_dir.name, "tts-output.mp3")
        response.stream_to_file(temp_filepath)

        await cl.Message(
            author=APP_NAME,
            content=f"You're hearing an AI voice generated by OpenAI's {model} model, using the {voice} style.  You can customize this in Settings if you'd like!",
            elements=[cl.Audio(name="", path=temp_filepath, display="inline")],
        ).send()

        __update_msg_history_from_assistant_with_ctx__(context)


def __update_msg_history_from_user_with_ctx__(context: str):
    __update_msg_history_with_ctx__(context=context, role="user")


def __update_msg_history_from_assistant_with_ctx__(context: str):
    __update_msg_history_with_ctx__(context=context, role="assistant")


def __update_msg_history_with_ctx__(context: str, role: str):
    if len(role) == 0 or len(context) == 0:
        return

    messages = cl.user_session.get("message_history") or []
    messages.append({"role": role, "content": context})


async def __handle_conversation(
    message: cl.Message, messages: List[Dict[str, str]]
) -> None:
    """
    Handles text-based conversations with the user.
    Routes the conversation based on settings and semantic understanding.
    """
    model = __get_settings(conf.SETTINGS_CHAT_MODEL)  # Get selected LLM model
    msg = cl.Message(content="", author=APP_NAME)
    await msg.send()

    query = message.content  # Get user query
    # Add query to message history
    __update_msg_history_from_user_with_ctx__(query)
    __update_msg_history_from_assistant_with_ctx__("Thinking...")

    if __is_currently_in_assistant_profile():
        mino = MinoAssistant()
        msg = cl.Message(content="", author=mino.name)
        await msg.send()

        res_message = mino.run_assistant(query)
        msg.content = res_message

        __update_msg_history_from_assistant_with_ctx__(res_message)

        enable_tts_response = __get_settings(conf.SETTINGS_ENABLE_TTS_RESPONSE)
        if enable_tts_response:
            msg.actions = [
                cl.Action(
                    name="speak_chat_response_action",
                    value=res_message,
                    label="Speak response",
                )
            ]

        await msg.update()

    else:
        use_dynamic_conversation_routing = __get_settings(
            conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING
        )

        if use_dynamic_conversation_routing:
            await __handle_dynamic_conversation_routing_chat__(
                messages, model, msg, query
            )
        else:
            await __handle_trigger_async_chat(
                llm_model=model, messages=messages, current_message=msg
            )


def __get_user_session_id__() -> str:
    return cl.user_session.get("id") or ""


def __get_settings(key: str) -> Any:
    """
    Retrieves a specific setting value from the user session.
    """
    settings = cl.user_session.get("chat_settings")
    if settings is None:
        return

    return settings[key]


async def __handle_vision__(
    input_image: str,
    prompt: str,
    is_local: bool = False,
) -> None:
    """
    Handles vision processing tasks using the specified vision model.
    Sends the processed image and description to the user.
    """
    vision_model = (
        conf.DEFAULT_VISION_MODEL
        if is_local
        else __get_settings(conf.SETTINGS_VISION_MODEL)
    )

    supports_vision = litellm.supports_vision(model=vision_model)

    if supports_vision is False:
        print(f"Unsupported vision model: {vision_model}")
        await cl.Message(
            content=f"It seems the vision model `{vision_model}` doesn't support image processing. Please choose a different model in Settings that offers Vision capabilities.",
        ).send()
        return

    message = cl.Message(
        content="I'm analyzing the image. This might take a moment.",
        author=APP_NAME,
    )

    await message.send()
    vresponse = await litellm.acompletion(
        user=__get_user_session_id__(),
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

    if is_local:
        image = cl.Image(path=input_image, name=prompt, display="inline")
    else:
        image = cl.Image(url=input_image, name=prompt, display="inline")

    message = cl.Message(
        author=APP_NAME,
        content="",
        elements=[
            image,
            cl.Text(name="Explain", content=description, display="inline"),
        ],
        actions=[
            cl.Action(
                name="speak_chat_response_action",
                value=description,
                label="Speak response",
            )
        ],
    )

    __update_msg_history_from_assistant_with_ctx__(description)

    await message.send()


async def __handle_trigger_async_chat(
    llm_model: str, messages: List[Dict[str, str]], current_message: cl.Message
) -> None:
    """
    Triggers an asynchronous chat completion using the specified LLM model.
    Streams the response back to the user and updates the message history.
    """

    temperature = __get_settings(conf.SETTINGS_TEMPERATURE)
    top_p = __get_settings(conf.SETTINGS_TOP_P)
    try:
        stream = await litellm.acompletion(
            model=llm_model,
            messages=messages,
            stream=True,
            num_retries=2,
            temperature=temperature,
            top_p=top_p,
        )

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await current_message.stream_token(token)

        content = current_message.content
        __update_msg_history_from_assistant_with_ctx__(content)

        enable_tts_response = __get_settings(conf.SETTINGS_ENABLE_TTS_RESPONSE)
        if enable_tts_response:
            current_message.actions = [
                cl.Action(
                    name="speak_chat_response_action",
                    value=content,
                    label="Speak response",
                )
            ]

        await current_message.update()

    except Exception as e:
        await __handle_exception_error(e)


async def __handle_exception_error(e: Exception) -> None:
    """
    Handles exceptions that occur during LLM interactions.
    """

    await cl.Message(
        content=(
            f"Something went wrong, please try again. Error type: {type(e)}, Error: {e}"
        )
    ).send()

    print(f"Error type: {type(e)}, Error: {e}")


async def __config_chat_session__(settings: Dict[str, Any]) -> None:
    """
    Configures the chat session based on user settings and sets the initial system message.
    """

    chat_profile = cl.user_session.get("chat_profile")
    if chat_profile == AppChatProfileType.CHAT.value:
        cl.user_session.set(
            conf.SETTINGS_CHAT_MODEL, settings[conf.SETTINGS_CHAT_MODEL]
        )

        system_message = {
            "role": "system",
            "content": "You are a helpful assistant who tries their best to answer questions: ",
        }

        cl.user_session.set("message_history", [system_message])

        msg = "Hello! I'm here to assist you. Please don't hesitate to ask me anything you'd like to know."
        await cl.Message(content=msg).send()
        __update_msg_history_from_assistant_with_ctx__(msg)

    elif chat_profile == AppChatProfileType.ASSISTANT.value:
        system_message = {"role": "system", "content": INSTRUCTIONS}

        cl.user_session.set("message_history", [system_message])

        msg = "Hello! I'm Mino, your Assistant. I'm here to assist you. Please don't hesitate to ask me anything you'd like to know. Currently, I can write and run code to answer math questions."

        await cl.Message(content=msg).send()
        __update_msg_history_from_assistant_with_ctx__(msg)


async def __handle_trigger_async_image_gen__(query: str) -> None:
    """
    Triggers asynchronous image generation using the default image generation model.
    Sends the generated image and description to the user.
    """
    image_gen_model = conf.DEFAULT_IMAGE_GEN_MODEL
    __update_msg_history_from_user_with_ctx__(query)

    message = cl.Message(
        content="Sure! I'll create an image based on your description. This might take a moment, please be patient.",
        author=APP_NAME,
    )
    await message.send()

    style = __get_settings(conf.SETTINGS_IMAGE_GEN_IMAGE_STYLE)
    quality = __get_settings(conf.SETTINGS_IMAGE_GEN_IMAGE_QUALITY)
    try:
        image_response = await litellm.aimage_generation(
            user=__get_user_session_id__(),
            prompt=query,
            model=image_gen_model,
            style=style,
            quality=quality,
        )

        image_gen_data = image_response["data"][0]
        image_url = image_gen_data["url"]
        revised_prompt = image_gen_data["revised_prompt"]

        message = cl.Message(
            author=APP_NAME,
            content="Here's the image, along with a refined description based on your input:",
            elements=[
                cl.Image(url=image_url, name=query, display="inline"),
                cl.Text(name="Description", content=revised_prompt, display="inline"),
            ],
            actions=[
                cl.Action(
                    name="speak_chat_response_action",
                    value=revised_prompt,
                    label="Speak response",
                )
            ],
        )

        __update_msg_history_from_assistant_with_ctx__(revised_prompt)

        await message.send()

    except Exception as e:
        await __handle_exception_error(e)


async def __handle_files_attachment__(
    message: cl.Message, messages: List[Dict[str, str]]
) -> None:
    """
    Handles file attachments from the user.
    Processes images using vision models and text files as chat input.
    """
    if not message.elements:
        await cl.Message(content="No file attached").send()
        return

    prompt = message.content
    __update_msg_history_from_user_with_ctx__(prompt)

    for file in message.elements:
        path = str(file.path)
        mime_type = file.mime or ""

        if "image" in mime_type:
            await __handle_vision__(path, prompt=prompt, is_local=True)

        elif "text" in mime_type:
            p = pathlib.Path(path, encoding="utf-8")
            s = p.read_text(encoding="utf-8")
            message.content = s
            await __handle_conversation(message, messages)

        elif "audio" in mime_type:
            f = pathlib.Path(path)
            await __handle_audio_transcribe__(path, f)


async def __handle_audio_transcribe__(path, audio_file):
    model = conf.DEFAULT_WHISPER_MODEL
    transcription = await async_openai_client.audio.transcriptions.create(
        model=model, file=audio_file
    )
    text = transcription.text

    await cl.Message(
        content="",
        author=APP_NAME,
        elements=[
            cl.Audio(name="Audio", path=path, display="inline"),
            cl.Text(content=text, name="Transcript", display="inline"),
        ],
    ).send()

    __update_msg_history_from_assistant_with_ctx__(text)
    return text


async def __handle_dynamic_conversation_routing_chat__(
    messages: List[Dict[str, str]], model: str, msg: cl.Message, query: str
) -> None:
    """
    Routes the conversation dynamically based on the semantic understanding of the user's query.
    Handles image generation, vision processing, and default chat interactions.
    """
    route_choice = route_layer(query)
    route_choice_name = route_choice.name

    should_trimmed_messages = __get_settings(conf.SETTINGS_TRIMMED_MESSAGES)
    if should_trimmed_messages:
        messages = trim_messages(messages, model)

    print(
        f"""💡
          Query: {query}
          Is classified as route: {route_choice_name}
          running router..."""
    )

    if route_choice_name == SemanticRouterType.IMAGE_GENERATION:
        print(
            f"""💡
            Running route_choice_name: {route_choice_name}.
            Processing image generation..."""
        )
        await __handle_trigger_async_image_gen__(query)

    elif route_choice_name == SemanticRouterType.VISION_IMAGE_PROCESSING:
        urls = extract_url(query)
        if len(urls) > 0:
            print(
                f"""💡
                Running route_choice_name: {route_choice_name}.
                Received image urls/paths.
                Processing with Vision model..."""
            )
            url = urls[0]
            await __handle_vision__(input_image=url, prompt=query, is_local=False)
        else:
            print(
                f"""💡
                Running route_choice_name: {route_choice_name}.
                Received no image urls/paths.
                Processing with async chat..."""
            )
            await __handle_trigger_async_chat(
                llm_model=model, messages=messages, current_message=msg
            )
    else:
        print(
            f"""💡
            Running route_choice_name: {route_choice_name}.
            Processing with async chat..."""
        )
        await __handle_trigger_async_chat(
            llm_model=model, messages=messages, current_message=msg
        )


def __is_currently_in_assistant_profile() -> bool:
    chat_profile = cl.user_session.get("chat_profile")
    return chat_profile == "Assistant"
