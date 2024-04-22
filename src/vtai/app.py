import os
import tempfile
from getpass import getpass
from typing import Any, Dict, List

import chainlit as cl
import dotenv
import litellm
import llms_config as conf
from chainlit.input_widget import Select, Switch
from constants import SemanticRouterType
from litellm.utils import trim_messages
from llm_profile_builder import build_llm_profile
from openai import OpenAI
from semantic_router.layer import RouteLayer
from url_extractor import extract_url

dotenv.load_dotenv()

# Model alias map for litellm
litellm.model_alias_map = conf.MODEL_ALIAS_MAP

# Load semantic router layer from JSON file
route_layer = RouteLayer.from_json("./src/vtai/semantic_route_layers.json")

# Initialize OpenAI client
openai_client = OpenAI(max_retries=2)

# Create temporary directory for TTS audio files
temp_dir = tempfile.TemporaryDirectory()

# Set OpenAI API Key from environment variable or user input
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
    "Enter OpenAI API Key: "
)


@cl.on_chat_start
async def start_chat():
    """
    Initializes the chat session.
    Builds LLM profiles, configures chat settings, and sets initial system message.
    """
    # build llm profile
    await build_llm_profile(conf.ICONS_PROVIDER_MAP)

    # settings configuration
    settings = await __build_settings()

    # set selected LLM model for current settion's model
    __config_chat_session(settings)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Handles incoming messages from the user.
    Processes text messages, file uploads, and routes conversations accordingly.
    """
    messages = cl.user_session.get("message_history") or []  # Get message history
    if len(message.elements) > 0:
        await __handle_files_upload(message, messages)  # Process file uploads
    else:
        await __handle_conversation(message, messages)  # Process text messages


async def __handle_conversation(
    message: cl.Message, messages: List[Dict[str, str]]
) -> None:
    """
    Handles text-based conversations with the user.
    Routes the conversation based on settings and semantic understanding.
    """
    model = __get_settings(conf.SETTINGS_CHAT_MODEL)  # Get selected LLM model
    msg = cl.Message(content="", author=model)  # Create initial response message
    await msg.send()

    query = message.content  # Get user query
    messages.append({"role": "user", "content": query})  # Add query to message history

    use_dynamic_conversation_routing = __get_settings(
        conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING
    )

    if use_dynamic_conversation_routing:
        await __handle_dynamic_conversation_routing(messages, model, msg, query)
    else:
        await __handle_trigger_async_chat(
            llm_model=model, messages=messages, current_message=msg
        )


@cl.on_settings_update
async def update_settings(settings: Dict[str, Any]) -> None:
    """
    Updates chat settings based on user preferences.
    """
    cl.user_session.set(conf.SETTINGS_CHAT_MODEL, settings[conf.SETTINGS_CHAT_MODEL])
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


async def __handle_trigger_async_chat(
    llm_model: str, messages: List[Dict[str, str]], current_message: cl.Message
) -> None:
    """
    Triggers an asynchronous chat completion using the specified LLM model.
    Streams the response back to the user and updates the message history.
    """
    try:
        stream = await litellm.acompletion(
            model=llm_model, messages=messages, stream=True, num_retries=2
        )
    except Exception as e:
        await __handle_exception_error(llm_model, messages, current_message, e)
        return

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await current_message.stream_token(token)

    content = current_message.content
    messages.append({"role": "assistant", "content": content})

    enable_tts_response = __get_settings(conf.SETTINGS_ENABLE_TTS_RESPONSE)
    if enable_tts_response:
        current_message.actions = [
            cl.Action(name="speak_response", value=content, label="Speak response")
        ]

    await current_message.update()


async def __handle_exception_error(
    llm_model: str,
    messages: List[Dict[str, str]],
    current_message: cl.Message,
    e: Exception,
) -> None:
    """
    Handles exceptions that occur during LLM interactions.
    """
    current_message.content = (
        f"Something went wrong, please try again. Error type: {type(e)}, Error: {e}"
    )
    print(f"Error type: {type(e)}, Error: {e}")


def __config_chat_session(settings: Dict[str, Any]) -> None:
    """
    Configures the chat session based on user settings and sets the initial system message.
    """
    cl.user_session.set(conf.SETTINGS_CHAT_MODEL, settings[conf.SETTINGS_CHAT_MODEL])

    system_message = {
        "role": "system",
        "content": "You are a helpful assistant who tries their best to answer questions: ",
    }

    cl.user_session.set("message_history", [system_message])


async def __build_settings() -> Dict[str, Any]:
    """
    Builds and sends chat settings to the user for configuration.
    """
    settings = await cl.ChatSettings(
        [
            Select(
                id=conf.SETTINGS_CHAT_MODEL,
                label="Chat Model",
                description="Select the Large Language Model (LLM) you want to use for chat conversations. Different models have varying strengths and capabilities.",
                values=conf.MODELS,
                initial_value=conf.DEFAULT_MODEL,
            ),
            Select(
                id=conf.SETTINGS_VISION_MODEL,
                label="Vision Model",
                description="Choose the vision model to analyze and understand images. This enables features like image description and object recognition.",
                values=conf.VISION_MODEL_MODELS,
                initial_value=conf.DEFAULT_VISION_MODEL,
            ),
            Switch(
                id=conf.SETTINGS_ENABLE_TTS_RESPONSE,
                label="Enable TTS",
                description=f"This feature allows you to hear the chat responses spoken aloud, which can be helpful for accessibility or multitasking. Note that this action requires an OpenAI API key. Default value is {conf.SETTINGS_ENABLE_TTS_RESPONSE_DEFAULT_VALUE}.",
                initial=conf.SETTINGS_ENABLE_TTS_RESPONSE_DEFAULT_VALUE,
            ),
            Select(
                id=conf.SETTINGS_TTS_MODEL,
                label="TTS Model",
                description="Select the TTS model to use for generating speech. Different models offer distinct voice styles and characteristics.",
                values=conf.TTS_MODEL_MODELS,
                initial_value=conf.DEFAULT_TTS_MODEL,
            ),
            Select(
                id=conf.SETTINGS_TTS_VOICE_PRESET_MODEL,
                label="TTS - Voice options",
                description="Choose the specific voice preset you prefer for TTS responses. Each preset offers a unique vocal style and tone.",
                values=conf.TTS_VOICE_PRESETS,
                initial_value=conf.DEFAULT_TTS_PRESET,
            ),
            Switch(
                id=conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING,
                label="Use dynamic conversation routing",
                description=f"This experimental feature automatically switches to specialized models based on your input. For example, if you ask to generate an image, it will use an image generation model like DALL·E 3. Note that this action requires an OpenAI API key. Default value is {conf.SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE}",
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


async def __handle_trigger_async_image_gen(
    messages: List[Dict[str, str]], query: str
) -> None:
    """
    Triggers asynchronous image generation using the default image generation model.
    Sends the generated image and description to the user.
    """
    image_gen_model = conf.DEFAULT_IMAGE_GEN_MODEL
    message = cl.Message(
        content=f"Sure! I'll use the `{image_gen_model}` model to create an image based on your description. This might take a moment, please be patient.",
        author=image_gen_model,
    )
    await message.send()

    try:
        image_response = await litellm.aimage_generation(
            prompt=query, model=image_gen_model
        )
    except Exception as e:
        await __handle_exception_error(image_gen_model, messages, message, e)
        return

    image_gen_data = image_response["data"][0]
    image_url = image_gen_data["url"]
    revised_prompt = image_gen_data["revised_prompt"]

    image = cl.Image(url=image_url, name=query, display="inline")
    revised_prompt_text = cl.Text(
        name="Description", content=revised_prompt, display="inline"
    )

    message = cl.Message(
        author=image_gen_model,
        content="Here's the image, along with a refined description based on your input:",
        elements=[image, revised_prompt_text],
    )

    await message.send()


async def __handle_files_upload(
    message: cl.Message, messages: List[Dict[str, str]]
) -> None:
    """
    Handles file uploads from the user.
    Processes images using vision models and text files as chat input.
    """
    if not message.elements:
        await cl.Message(content="No file attached").send()
        return

    prompt = message.content

    for file in message.elements:
        if "image" in file.mime:
            path = file.path
            with open(path, "r") as f:
                pass
            await __handle_vision(path, prompt=prompt, messages=messages, is_local=True)
            break

        elif "text" in file.mime:
            with open(file.path, "r") as uploaded_file:
                content = uploaded_file.read()
            messages.append({"role": "user", "content": content})
            await __handle_conversation(message, messages)
            break

        else:
            await cl.Message(
                content=f"Hmm, we can't handle the file `{file.name}` just yet. How about trying a JPEG, PNG, or a simple text file instead?"
            ).send()


async def __handle_dynamic_conversation_routing(
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
        await __handle_trigger_async_image_gen(messages, query)

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
            await __handle_vision(
                input_image=url, prompt=query, messages=messages, is_local=False
            )
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


async def __handle_vision(
    input_image: str,
    prompt: str,
    messages: List[Dict[str, str]],
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
        content=f"Analyzing the image using the `{vision_model}` model... This might take a moment. 🔎",
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
    messages.append({"role": "user", "content": prompt})

    if is_local:
        image = cl.Image(path=input_image, name=prompt, display="inline")
    else:
        image = cl.Image(url=input_image, name=prompt, display="inline")

    revised_prompt_text = cl.Text(name="Explain", content=description, display="inline")
    message = cl.Message(
        author=vision_model, content="", elements=[image, revised_prompt_text]
    )

    await message.send()


def __get_settings(key: str) -> Any:
    """
    Retrieves a specific setting value from the user session.
    """
    return cl.user_session.get("chat_settings")[key]


@cl.action_callback("speak_response")
async def on_action(action: cl.Action) -> None:
    """
    Handles the "speak_response" action triggered by the user.
    """
    await action.remove()
    value = action.value
    return await __handle_tts_response(value)


async def __handle_tts_response(value: str) -> None:
    """
    Generates and sends a TTS audio response using OpenAI's Audio API.
    """
    enable_tts_response = __get_settings(conf.SETTINGS_ENABLE_TTS_RESPONSE)
    if enable_tts_response is False:
        return

    if len(value) == 0:
        return

    model = __get_settings(conf.SETTINGS_TTS_MODEL)
    voice = __get_settings(conf.SETTINGS_TTS_VOICE_PRESET_MODEL)

    with openai_client.audio.speech.with_streaming_response.create(
        model=model, voice=voice, input=value, response_format="aac"
    ) as response:
        temp_filepath = os.path.join(temp_dir.name, "tts-output.m4a")
        response.stream_to_file(temp_filepath)

        await cl.Message(
            author=model,
            content=f"You're hearing an AI voice generated by OpenAI's {model} model, using the {voice} style.  You can customize this in Settings if you'd like!",
            elements=[cl.Audio(name="", path=temp_filepath, display="inline")],
        ).send()
