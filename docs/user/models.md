# Models

This page provides information about the AI models supported by VT.ai and how to use them effectively.

## Supported Models

VT.ai integrates with multiple AI providers and supports a wide range of models:

### OpenAI Models

- **GPT-o1** (o1): High-performance general purpose model
- **GPT-o1 Mini** (o1-mini): Compact version of GPT-o1
- **GPT-o1 Pro** (o1-pro): Enhanced version with advanced capabilities
- **GPT-o3 Mini** (o3-mini): Compact, efficient model for everyday tasks
- **GPT-4.5 Preview** (gpt-4.5-preview): Preview of next-generation capabilities
- **GPT-4o** (4o): Advanced vision and multimodal capabilities

### Anthropic Models

- **Claude 3.5 Sonnet** (c3.5-sonnet): Balanced performance and efficiency
- **Claude 3.7 Sonnet** (sonnet): Advanced reasoning capabilities
- **Claude 3.5 Haiku** (c3.5-haiku): Fast, efficient model for common tasks
- **Claude 3 Opus** (opus): Highest capability model for complex tasks

### Google Models

- **Gemini 1.5 Pro** (gemini-1.5-pro): Advanced multimodal capabilities
- **Gemini 1.5 Flash** (gemini-1.5-flash): Fast, efficient model
- **Gemini 2.0** (gemini-2.0): Advanced reasoning model
- **Gemini 2.5 Pro** (gemini-2.5-pro): Latest Google model with enhanced capabilities
- **Gemini 2.5 Flash** (gemini-2.5-flash): Fast version of Gemini 2.5

### DeepSeek Models

- **DeepSeek-Coder** (deepseek-coder): Specialized for coding tasks
- **DeepSeek Chat** (deepseek-chat): General conversation model
- **DeepSeek R1 Series** (deepseek-r1): Next-generation reasoning models (multiple sizes)

### Groq Models

- **Llama 4 Scout 17b Instruct**: Fast inference of Llama 4 Scout via Groq
- **Llama 3 8b/70b**: Optimized versions of Llama 3 on Groq's infrastructure
- **Mixtral 8x7b**: Fast inference of Mixtral model

### Cohere Models

- **Command**: General purpose instruction model
- **Command-R**: Enhanced reasoning capabilities
- **Command-Light**: Lightweight, efficient model
- **Command-R-Plus**: Advanced reasoning with extended capabilities

### OpenRouter Integration

VT.ai supports many models through OpenRouter, including:

- **Qwen Models**: Qwen 2.5 VL 32B, Qwen 2.5 Coder 32B, etc.
- **Mistral Models**: Mistral Small 3.1 24B and others
- **Additional proprietary and open models**

### Local Models (via Ollama)

Run models locally for privacy and offline use:

- **Llama 3**: Multiple sizes (8B, 70B)
- **DeepSeek R1**: Various sizes (1.5B, 7B, 8B, 14B, 32B, 70B)
- **Qwen2.5-coder**: Multiple versions (7b, 14b, 32b)
- **Mistral**: Various versions
- **Many other open source models**

## Model Selection

You can select models in several ways:

1. **Command Line**:

   ```bash
   vtai --model sonnet
   ```

2. **UI Settings**:
   - Use the model selector in the settings menu
   - Change models during a conversation

3. **Dynamic Routing**:
   - Allow VT.ai to automatically select the best model for your query
   - Enable in settings with "Use Dynamic Conversation Routing"

## Model Capabilities

Different models have different capabilities:

### Vision-Capable Models

For analyzing images and visual content:

- GPT-4o
- Gemini 1.5 Pro/Flash
- Gemini 2.5 Pro/Flash
- Claude 3 Sonnet/Opus
- Llama 3.2 Vision
- Qwen 2.5 VL

### TTS-Capable Models

For text-to-speech generation:

- GPT-4o mini TTS
- TTS-1
- TTS-1-HD
- Various voice options: alloy, echo, fable, onyx, nova, shimmer

### Image Generation Models

For creating images from text descriptions:

- **DALL-E 3**: OpenAI's image generation model
- **GPT-Image-1**: Advanced image generation model with customizable settings
  - Supports transparent backgrounds
  - Multiple output formats (PNG, JPEG, WEBP)
  - Customizable dimensions (square, landscape, portrait)
  - Quality settings (standard, high)
  - Advanced compression options (0-100 for webp/jpeg)
  - HD options for higher quality outputs

### Reasoning-Enhanced Models

VT.ai supports special "thinking mode" with these models:

- DeepSeek Reasoner
- DeepSeek R1 series
- Qwen 2.5 models
- Claude 3 models
- GPT-4o

## Performance Considerations

When choosing models, consider these factors:

- **Speed**: Models like GPT-o3 Mini, Groq-accelerated models, and Claude 3 Haiku are faster
- **Quality**: Models like GPT-o1, GPT-4o, and Claude 3 Opus offer higher quality
- **Cost**: Smaller models generally cost less to use
- **Multimodal Needs**: Only some models support image analysis
- **Local Computation**: Ollama models run locally but require more resources
- **API Availability**: Some models may require specific API keys

## API Key Configuration

For most models, you'll need to configure the appropriate API keys:

```bash
# OpenAI models (o1, o3-mini, 4o, etc.)
export OPENAI_API_KEY="sk-your-key-here"

# Anthropic models (sonnet, opus, etc.)
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Google models (gemini series)
export GEMINI_API_KEY="your-key-here"

# DeepSeek models
export DEEPSEEK_API_KEY="your-key-here"

# Groq models
export GROQ_API_KEY="your-key-here"

# Cohere models
export COHERE_API_KEY="your-key-here"

# OpenRouter (for access to multiple providers)
export OPENROUTER_API_KEY="your-key-here"
```

You can also set these keys when starting VT.ai:

```bash
vtai --api-key openai=sk-your-key-here
```

For more details on configuration, see the [Configuration](configuration.md) page.
