from chainlit.types import ChatProfile

from utils.chat_profile import AppChatProfileModel, AppChatProfileType

# settings
SETTINGS_CHAT_MODEL = "settings_chat_model"
SETTINGS_VISION_MODEL = "settings_vision_model"
SETTINGS_IMAGE_GEN_LLM_MODEL = "settings_image_gen_llm_model"
SETTINGS_IMAGE_GEN_IMAGE_STYLE = "settings_image_gen_image_style"
SETTINGS_IMAGE_GEN_IMAGE_QUALITY = "settings_image_gen_image_quality"
SETTINGS_TTS_MODEL = "settings_tts_model"
SETTINGS_TTS_VOICE_PRESET_MODEL = "settings_tts_voice_preset_model"
SETTINGS_ENABLE_TTS_RESPONSE = "settings_enable_tts_response"

SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING = "settings_use_dynamic_conversation_routing"
SETTINGS_TRIMMED_MESSAGES = "settings_trimmed_messages"
SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE = True
SETTINGS_TRIMMED_MESSAGES_DEFAULT_VALUE = True
SETTINGS_ENABLE_TTS_RESPONSE_DEFAULT_VALUE = True

# ref https://platform.openai.com/docs/api-reference/chat
SETTINGS_TEMPERATURE = "settings_temperature"
DEFAULT_TEMPERATURE = 0
SETTINGS_TOP_K = "settings_top_k"
SETTINGS_TOP_P = "settings_top_p"
DEFAULT_TOP_P = 1

# set models alias mapping

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_IMAGE_GEN_MODEL = "dall-e-3"
DEFAULT_VISION_MODEL = "gemini/gemini-pro-vision"
DEFAULT_TTS_MODEL = "tts-1"
DEFAULT_TTS_PRESET = "nova"
DEFAULT_WHISPER_MODEL = "whisper-1"

SETTINGS_IMAGE_GEN_IMAGE_STYLES = ["vivid", "natural"]
SETTINGS_IMAGE_GEN_IMAGE_QUALITIES = ["standard", "hd"]

TTS_MODELS_MAP = {
    "OpenAI - Text-to-speech 1": "tts-1",
    "OpenAI - Text-to-speech 1 HD": "tts-1-hd",
}

TTS_VOICE_PRESETS = [
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
]

IMAGE_GEN_MODELS_ALIAS_MAP = {
    "OpenAI - DALL·E 3": "dall-e-3",
}

VISION_MODEL_MAP = {
    "OpenAI - GPT 4 Turbo": "gpt-4-turbo",
    "Google - Gemini 1.5 Pro": "gemini/gemini-1.5-pro-latest",
    "Google - Gemini Pro Vision": "gemini/gemini-pro-vision",
}

MODEL_ALIAS_MAP = {
    "OpenAI - GPT 3.5 Turbo": "gpt-3.5-turbo",
    "OpenAI - GPT 4 Turbo": "gpt-4-turbo",
    "OpenAI - GPT 4": "gpt-4",
    "Groq - Llama 3 8b": "groq/llama3-8b-8192",
    "Groq - Llama 3 70b": "groq/llama3-70b-8192",
    "Groq - Mixtral 8x7b": "groq/mixtral-8x7b-32768",
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
    "VT.ai": "./src/resources/vt.jpg",
    "Mino": "./src/resources/vt.jpg",
    "tts-1": "./src/resources/chatgpt-icon.png",
    "tts-1-hd": "./src/resources/chatgpt-icon.png",
    "OpenAI": "./src/resources/chatgpt-icon.png",
    "Ollama": "./src/resources/ollama.png",
    "Anthropic": "./src/resources/claude-ai-icon.png",
    "OpenRouter" "Google": "./src/resources/google-gemini-icon.png",
    "Groq": "./src/resources/groq.ico",
    "gemini/gemini-pro-vision": "./src/resources/google-gemini-icon.png",
    "dall-e-3": "./src/resources/chatgpt-icon.png",
    "gpt-4": "./src/resources/chatgpt-icon.png",
    "gpt-4-turbo": "./src/resources/chatgpt-icon.png",
    "gpt-3.5-turbo": "./src/resources/chatgpt-icon.png",
    "command": "./src/resources/cohere.ico",
    "command-r": "./src/resources/cohere.ico",
    "command-light": "./src/resources/cohere.ico",
    "command-r-plus": "./src/resources/cohere.ico",
    "claude-2": "./src/resources/claude-ai-icon.png",
    "claude-3-sonnet-20240229": "./src/resources/claude-ai-icon.png",
    "claude-3-haiku-20240307": "./src/resources/claude-ai-icon.png",
    "claude-3-opus-20240229": "./src/resources/claude-ai-icon.png",
    "groq/llama3-8b-8192": "./src/resources/groq.ico",
    "groq/llama3-70b-8192": "./src/resources/groq.ico",
    "groq/mixtral-8x7b-32768": "./src/resources/groq.ico",
    "gemini/gemini-1.5-pro-latest": "./src/resources/google-gemini-icon.png",
    "openrouter/mistralai/mistral-7b-instruct": "./src/resources/openrouter.ico",
    "OpenRouter - Mistral 7b instruct Free": "./src/resources/openrouter.ico",
    "ollama/llama3": "./src/resources/ollama.png",
    "ollama/mistral": "./src/resources/ollama.png",
}

NAMES = list(MODEL_ALIAS_MAP.keys())
MODELS = list(MODEL_ALIAS_MAP.values())

IMAGE_GEN_NAMES = list(IMAGE_GEN_MODELS_ALIAS_MAP.keys())
IMAGE_GEN_MODELS = list(IMAGE_GEN_MODELS_ALIAS_MAP.values())

VISION_MODEL_NAMES = list(VISION_MODEL_MAP.keys())
VISION_MODEL_MODELS = list(VISION_MODEL_MAP.values())

TTS_MODEL_NAMES = list(TTS_MODELS_MAP.keys())
TTS_MODEL_MODELS = list(TTS_MODELS_MAP.values())

APP_CHAT_PROFILE_CHAT = AppChatProfileModel(
    title=AppChatProfileType.CHAT.value,
    description="Multi-modal chat with LLM.",
)

APP_CHAT_PROFILE_ASSISTANT = AppChatProfileModel(
    title=AppChatProfileType.ASSISTANT.value,
    description="[Beta] Use Mino built-in Assistant to ask complex question. Currently support Math Calculator",
)

APP_CHAT_PROFILES = [APP_CHAT_PROFILE_CHAT, APP_CHAT_PROFILE_ASSISTANT]

CHAT_PROFILES = [
    ChatProfile(
        name=profile.title,
        markdown_description=profile.description,
    )
    for profile in APP_CHAT_PROFILES
]
