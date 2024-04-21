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
