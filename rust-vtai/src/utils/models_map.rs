use std::collections::HashMap;
use once_cell::sync::Lazy;

// Model mappings (based on llm_providers_config.py)
pub static MODEL_ALIAS_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    // DeepSeek models
    map.insert("DeepSeek R1", "deepseek/deepseek-reasoner");
    map.insert("DeepSeek V3", "deepseek/deepseek-chat");
    map.insert("DeepSeek Coder", "deepseek/deepseek-coder");
    // OpenAI models
    map.insert("OpenAI - GPT-4.1", "gpt-4.1");
    map.insert("OpenAI - GPT-4.1 Mini", "gpt-4.1-mini");
    map.insert("OpenAI - GPT-4.1 Nano", "gpt-4.1-nano");
    map.insert("OpenAI - GPT-4o Mini", "gpt-4o-mini");
    map.insert("OpenAI - GPT-4o", "gpt-4o");
    map.insert("OpenAI - GPT-o1", "o1");
    map.insert("OpenAI - GPT-4.5 Preview", "gpt-4.5-preview");
    map.insert("OpenAI - GPT-o3 Mini", "o3-mini");
    map.insert("OpenAI - GPT-o1 Mini", "o1-mini");
    map.insert("OpenAI - GPT-o1 Pro", "o1-pro");
    // Anthropic models
    map.insert("Anthropic - Claude 3.7 Sonnet", "claude-3-7-sonnet-20250219");
    map.insert("Anthropic - Claude 3.5 Sonnet", "claude-3-5-sonnet-20241022");
    map.insert("Anthropic - Claude 3.5 Haiku", "claude-3-5-haiku-20241022");
    // Google models
    map.insert("Google - Gemini 2.0 Pro", "gemini/gemini-2.0-pro");
    map.insert("Google - Gemini 2.0 Flash", "gemini/gemini-2.0-flash");
    map.insert("Google - Gemini 2.0 Flash Exp", "gemini/gemini-2.0-flash-exp");
    // OpenRouter models
    map.insert("OpenRouter - DeepSeek R1 (free)", "openrouter/deepseek/deepseek-r1:free");
    map.insert("OpenRouter - DeepSeek R1", "openrouter/deepseek/deepseek-r1");
    map.insert("OpenRouter - DeepSeek V3 0324 (free)", "openrouter/deepseek/deepseek-chat-v3-0324:free");
    map.insert("OpenRouter - DeepSeek V3 0324", "openrouter/deepseek/deepseek-chat-v3-0324");
    map.insert("OpenRouter - Anthropic: Claude 3.7 Sonnet (thinking)", "openrouter/anthropic/claude-3.7-sonnet:thinking");
    map.insert("OpenRouter - Anthropic: Claude 3.7 Sonnet", "openrouter/anthropic/claude-3.7-sonnet");
    map.insert("OpenRouter - Google: Gemini 2.5 Pro Experimental (free)", "openrouter/google/gemini-2.5-pro-exp-03-25:free");
    map.insert("OpenRouter - Google: Gemini 2.5 Pro Preview", "openrouter/google/gemini-2.5-pro-preview-03-25");
    map.insert("OpenRouter - Google: Gemini 2.0 Flash Thinking Experimental (free)", "openrouter/google/gemini-2.0-flash-thinking-exp:free");
    map.insert("OpenRouter - Google: Gemini 2.0 Flash Experimental (free)", "openrouter/google/gemini-2.0-flash-exp:free");
    map.insert("OpenRouter - Google: Gemma 3 27B (free)", "openrouter/google/gemma-3-27b-it:free");
    map.insert("OpenRouter - Meta: Llama 4 Maverick (free)", "openrouter/meta-llama/llama-4-maverick:free");
    map.insert("OpenRouter - Meta: Llama 4 Maverick", "openrouter/meta-llama/llama-4-maverick");
    map.insert("OpenRouter - Meta: Llama 4 Scout (free)", "openrouter/meta-llama/llama-4-scout:free");
    map.insert("OpenRouter - Meta: Llama 4 Scout", "openrouter/meta-llama/llama-4-scout");
    map.insert("OpenRouter - Qwen QWQ 32B (free)", "openrouter/qwen/qwq-32b:free");
    map.insert("OpenRouter - Qwen QWQ 32B", "openrouter/qwen/qwq-32b");
    map.insert("OpenRouter - Qwen 2.5 VL 32B (free)", "openrouter/qwen/qwen2.5-vl-32b-instruct:free");
    map.insert("OpenRouter - Qwen 2.5 Coder 32B", "openrouter/qwen/qwen-2.5-coder-32b-instruct");
    map.insert("OpenRouter - Mistral: Mistral Small 3.1 24B (free)", "openrouter/mistralai/mistral-small-3.1-24b-instruct:free");
    map.insert("OpenRouter - Mistral: Mistral Small 3.1 24B", "openrouter/mistralai/mistral-small-3.1-24b-instruct");
    // Ollama models
    map.insert("Ollama - Deepseek R1 1.5B", "ollama/deepseek-r1:1.5b");
    map.insert("Ollama - Deepseek R1 7B", "ollama/deepseek-r1:7b");
    map.insert("Ollama - Deepseek R1 8B", "ollama/deepseek-r1:8b");
    map.insert("Ollama - Deepseek R1 14B", "ollama/deepseek-r1:14b");
    map.insert("Ollama - Deepseek R1 32B", "ollama/deepseek-r1:32b");
    map.insert("Ollama - Deepseek R1 70B", "ollama/deepseek-r1:70b");
    map.insert("Ollama - Qwen2.5-coder 7b", "ollama/qwen2.5-coder");
    map.insert("Ollama - Qwen2.5-coder 14b", "ollama/qwen2.5-coder:14b");
    map.insert("Ollama - Qwen2.5-coder 32b", "ollama/qwen2.5-coder:32b");
    map.insert("Ollama - LLama 3.2 Vision", "ollama/llama3.2-vision");
    map.insert("Ollama - LLama 3", "ollama/llama3");
    map.insert("Ollama - LLama 3.1", "ollama/llama3.1");
    map.insert("Ollama - LLama 3.2 - Mini", "ollama/llama3.2");
    map.insert("Ollama - Phi-3", "ollama/phi3");
    map.insert("Ollama - Command R", "ollama/command-r");
    map.insert("Ollama - Command R+", "ollama/command-r-plus");
    map.insert("Ollama - Mistral 7B Instruct", "ollama/mistral");
    map.insert("Ollama - Mixtral 8x7B Instruct", "ollama/mixtral");
    // Mistral
    map.insert("Mistral Small", "mistral/mistral-small-latest");
    map.insert("Mistral Large", "mistral/mistral-large-latest");
    // Groq models
    map.insert("Groq - Llama 4 Scout 17b Instruct", "meta-llama/llama-4-scout-17b-16e-instruct");
    map.insert("Groq - Llama 3 8b", "groq/llama3-8b-8192");
    map.insert("Groq - Llama 3 70b", "groq/llama3-70b-8192");
    map.insert("Groq - Mixtral 8x7b", "groq/mixtral-8x7b-32768");
    // Cohere models
    map.insert("Cohere - Command", "command");
    map.insert("Cohere - Command-R", "command-r");
    map.insert("Cohere - Command-Light", "command-light");
    map.insert("Cohere - Command-R-Plus", "command-r-plus");
    map
});

pub static VISION_MODEL_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("OpenAI - GPT-4o", "gpt-4o");
    map.insert("OpenAI - GPT 4 Turbo", "gpt-4-turbo");
    map.insert("Google - Gemini 1.5 Flash", "gemini/gemini-2.0-flash");
    map.insert("Google - Gemini 1.5 Pro", "gemini/gemini-2.0-pro");
    map.insert("Ollama - LLama 3.2 Vision", "ollama/llama3.2-vision");
    map
});

pub static IMAGE_GEN_MODELS_ALIAS_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("OpenAI - DALL·E 3", "dall-e-3");
    map
});

pub static TTS_MODELS_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("OpenAI - GPT-4o mini TTS", "gpt-4o-mini-tts");
    map.insert("OpenAI - Text-to-speech 1", "tts-1");
    map.insert("OpenAI - Text-to-speech 1 HD", "tts-1-hd");
    map
});

// List of models that benefit from <think> tag for reasoning
pub static REASONING_MODELS: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "deepseek/deepseek-reasoner",
        "openrouter/deepseek/deepseek-r1:free",
        "openrouter/deepseek/deepseek-r1",
        "openrouter/deepseek/deepseek-chat-v3-0324:free",
        "openrouter/deepseek/deepseek-chat-v3-0324",
        "ollama/deepseek-r1:1.5b",
        "ollama/deepseek-r1:7b",
        "ollama/deepseek-r1:8b",
        "ollama/deepseek-r1:14b",
        "ollama/deepseek-r1:32b",
        "ollama/deepseek-r1:70b",
        // Add other reasoning models from Python list here
        "openrouter/anthropic/claude-3.7-sonnet:thinking",
        "openrouter/google/gemini-2.0-flash-thinking-exp:free",
    ]
});

/// Resolve a model alias to its actual ID
pub fn resolve_model_alias(alias: &str) -> Option<&'static str> {
    MODEL_ALIAS_MAP.get(alias).copied()
}

/// Check if a model is a reasoning model
pub fn is_reasoning_model(model_id: &str) -> bool {
    REASONING_MODELS.iter().any(|&reasoning_model| model_id.contains(reasoning_model))
}
