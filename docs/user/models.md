# Models

This page provides information about the AI models supported by VT.ai and how to use them effectively.

## Supported Models

VT.ai integrates with multiple AI providers and supports a wide range of models:

### OpenAI Models

- **GPT-o1** (o1): High-performance general purpose model
- **GPT-o3 Mini** (o3-mini): Compact, efficient model for everyday tasks
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

### Other Providers

- **DeepSeek** (deepseek): High-quality alternative models
- **Cohere** (cohere): Specialized language models
- **Mistral** (mistral): Efficient open models
- **Llama 3** (llama3): Meta's advanced open model
- **Local Models** (via Ollama): Run models locally for privacy and offline use

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
- Claude 3 Sonnet/Opus
- Llama 3.2 Vision

### TTS-Capable Models

For text-to-speech generation:

- GPT-4o mini TTS
- TTS-1
- TTS-1-HD

### Image Generation Models

For creating images from text descriptions:

- DALL-E 3

## Performance Considerations

When choosing models, consider these factors:

- **Speed**: Models like GPT-o3 Mini and Claude 3 Haiku are faster
- **Quality**: Models like GPT-o1, GPT-4o, and Claude 3 Opus offer higher quality
- **Cost**: Smaller models generally cost less to use
- **Multimodal Needs**: Only some models support image analysis
- **Local Computation**: Ollama models run locally but require more resources

*This page is under construction. More detailed model information will be added soon.*
