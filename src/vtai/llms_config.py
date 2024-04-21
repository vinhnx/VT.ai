# settings
SETTINGS_CHAT_MODEL = "settings_chat_model"
SETTINGS_VISION_MODEL = "settings_vision_model"
SETTINGS_IMAGE_GEN_LLM_MODEL = "settings_image_gen_llm_model"
SETTINGS_LLM_PARAMS_TEMPERATURE = "settings_temperature"
SETTINGS_LLM_PARAMS_TOP_P = "settings_top_p"
SETTINGS_LLM_PARAMS_MAX_TOKENS = "settings_max_tokens"
SETTINGS_LLM_PARAMS_STOP_SEQUENCE = "settings_stop_sequence"
SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING = "settings_use_dynamic_conversation_routing"
SETTINGS_TRIMMED_MESSAGES = "settings_trimmed_messages"
SETTINGS_USE_DYNAMIC_CONVERSATION_ROUTING_DEFAULT_VALUE = True
SETTINGS_TRIMMED_MESSAGES_DEFAULT_VALUE = True

# set models alias mapping

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_IMAGE_GEN_MODEL = "dall-e-3"
DEFAULT_VISION_MODEL = "gemini/gemini-pro-vision"

IMAGE_GEN_MODELS_ALIAS_MAP = {
    "OpenAI - DALL·E 3": "dall-e-3",
}

VISION_MODEL_MAP = {
    "OpenAI - GPT 4 Turbo": "gpt-4-turbo",
    "OpenAI - GPT 4 Vision Preview": "gpt-4-vision-preview",
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
    "gemini/gemini-pro-vision": "./src/vtai/resources/google-gemini-icon.png",
    "gpt-4-vision-preview": "./src/vtai/resources/chatgpt-icon.png",
    "dall-e-3": "./src/vtai/resources/chatgpt-icon.png",
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

IMAGE_GEN_NAMES = list(IMAGE_GEN_MODELS_ALIAS_MAP.keys())
IMAGE_GEN_MODELS = list(IMAGE_GEN_MODELS_ALIAS_MAP.values())

VISION_MODEL_NAMES = list(VISION_MODEL_MAP.keys())
VISION_MODEL_MODELS = list(VISION_MODEL_MAP.values())
