import chainlit as cl
from chainlit.input_widget import Select
from llm_profile_builder import build_llm_profile

import litellm

# keys
LLM_MODEL_USER_SESSION_KEY = "llm_model"

# set models alias mapping
DEFAULT_MODEL = "openrouter/mistralai/mistral-7b-instruct:free"
MODEL_ALIAS_MAP = {
    "Groq - Llama 3 8b": "groq/llama3-8b-8192",
    "Groq - Llama 3 70b": "groq/llama3-70b-8192",
    "Groq - Mixtral 8x7b": "groq/mixtral-8x7b-32768",
    "OpenAI - GPT 3.5 Turbo": "gpt-3.5-turbo",
    "OpenAI - GPT 4 Turbo": "gpt-4-turbo",
    "OpenAI - GPT 4": "gpt-4",
    "Cohere - Command": "command",
    "Cohere - Command-R": "command-r",
    "Cohere - Command-Light": "command-light",
    "Cohere - Command-R-Plus": "command-r-plus",
    "Google - Gemini Pro": "gemini/gemini-1.5-pro-latest",
    "OpenRouter - Mistral 7b instruct": "openrouter/mistralai/mistral-7b-instruct",
    "OpenRouter - Mistral 7b instruct Free": "openrouter/mistralai/mistral-7b-instruct:free",
    "Ollama - LLama 3": "ollama/llama3",
    "Ollama - Mistral": "ollama/mistral",
    "Anthropic - Claude 3 Sonnet": "claude-3-sonnet-20240229",
    "Anthropic - Claude 3 Haiku": "claude-3-haiku-20240307",
    "Anthropic - Claude 3 Opus": "claude-3-opus-20240229",
}

ICONS_PROVIDER_MAP = {
    "gpt-4": "./src/vtai/resources/chatgpt-icon.png",
    "gpt-4-turbo": "./src/vtai/resources/chatgpt-icon.png",
    "gpt-3.5-turbo": "./src/vtai/resources/chatgpt-icon.png",
    "command": "./src/vtai/resources/cohere.ico",
    "command-r": "./src/vtai/resources/cohere.ico",
    "command-light": "./src/vtai/resources/cohere.ico",
    "command-r-plus": "./src/vtai/resources/cohere.ico",
    "claude-2": "./src/vtai/resources/claude-ai-icon.png",
    "claude-3-sonnet-20240229": "./src/vtai/resources/claude-ai-icon.png",
    "claude-3-haiku-20240307": "./src/vtai/resources/claude-ai-icon.png",
    "claude-3-opus-20240229": "./src/vtai/resources/claude-ai-icon.png",
    "groq/llama3-8b-8192": "./src/vtai/resources/groq.ico",
    "groq/llama3-70b-8192": "./src/vtai/resources/groq.ico",
    "groq/mixtral-8x7b-32768": "./src/vtai/resources/groq.ico",
    "gemini/gemini-1.5-pro-latest": "./src/vtai/resources/google-gemini-icon.png",
    "openrouter/mistralai/mistral-7b-instruct": "./src/vtai/resources/openrouter.ico",
    "OpenRouter - Mistral 7b instruct Free": "./src/vtai/resources/openrouter.ico",
    "ollama/llama3": "./src/vtai/resources/ollama.png",
    "ollama/mistral": "./src/vtai/resources/ollama.png",
}

NAMES = list(MODEL_ALIAS_MAP.keys())
MODELS = list(MODEL_ALIAS_MAP.values())

litellm.model_alias_map = MODEL_ALIAS_MAP


@cl.on_chat_start
async def start_chat():
    # build llm profile
    await build_llm_profile(ICONS_PROVIDER_MAP)

    # settings configuration
    settings = await cl.ChatSettings(
        [
            Select(
                id=LLM_MODEL_USER_SESSION_KEY,
                label="Choose LLM model",
                values=MODELS,
                initial_index=0,
            )
        ]
    ).send()

    # set selected LLM model for current settion's model
    cl.user_session.set(
        LLM_MODEL_USER_SESSION_KEY, settings[LLM_MODEL_USER_SESSION_KEY]
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

    model = str(cl.user_session.get(LLM_MODEL_USER_SESSION_KEY) or DEFAULT_MODEL)
    msg = cl.Message(content="", author=model)
    await msg.send()

    messages.append(
        {
            "role": "user",
            "content": message.content,
        }
    )

    # trigger async litellm model with message
    # with streaming
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
    llm_model = settings[LLM_MODEL_USER_SESSION_KEY]
    cl.user_session.set(LLM_MODEL_USER_SESSION_KEY, llm_model)
