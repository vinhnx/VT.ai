# Models Integration

This guide explains how VT.ai integrates with different AI model providers and how to work with the models API.

## Architecture Overview

VT.ai uses LiteLLM as a unified interface to multiple AI providers, which provides:

- A consistent API across different models
- Automatic fallbacks and retries
- Standardized error handling
- Easy switching between models

## Provider Integration

### Built-in Providers

VT.ai comes with built-in support for several providers:

- OpenAI (GPT-o1, GPT-o3, GPT-4o)
- Anthropic (Claude models)
- Google (Gemini models)
- Local models via Ollama
- And others (DeepSeek, Cohere, etc.)

### Provider Configuration

Provider configuration is managed in `vtai/utils/llm_providers_config.py`, where:

- Model-to-provider mappings are defined
- Environment variable names for API keys are specified
- Default parameters for each model are set
- Icons and display names are configured

## Working with Models

### Model Selection

Models are selected through several mechanisms:

1. **User Selection**: Via UI or command line
2. **Semantic Router**: Automatically based on query
3. **Specialized Handlers**: For specific tasks (vision, image generation)

### Model Configuration

Models can be configured with parameters like:

```python
# Example configuration
model_params = {
    "model": "o3-mini",  # OpenAI GPT-o3 Mini model
    "temperature": 0.7,  # Controls randomness
    "top_p": 0.9,        # Controls diversity
    "max_tokens": 1000   # Maximum output length
}
```

### Calling Models

VT.ai uses LiteLLM's completion interface for consistency:

```python
# Example asynchronous call using LiteLLM
from litellm import acompletion

async def call_model(messages, model="o3-mini", **kwargs):
    try:
        response = await acompletion(
            model=model,
            messages=messages,
            **kwargs
        )
        return response
    except Exception as e:
        # Error handling
        raise
```

## Specialized Model Usage

### Vision Models

Vision models require special handling for image inputs:

```python
# Example vision model call
async def call_vision_model(image_url, prompt, model="4o", **kwargs):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]

    response = await acompletion(
        model=model,
        messages=messages,
        **kwargs
    )
    return response
```

### TTS Models

Text-to-speech models handle audio generation:

```python
# Example TTS call
async def generate_speech(text, model="tts-1", voice="alloy"):
    from litellm import atts

    try:
        response = await atts(
            text=text,
            model=model,
            voice=voice
        )
        return response
    except Exception as e:
        # Error handling
        raise
```

### Image Generation Models

VT.ai now uses GPT-Image-1 for image generation with advanced configuration options:

```python
# Example image generation call
async def generate_image(prompt, **kwargs):
    # GPT-Image-1 is now the default image generation model
    response = await client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=1,
        size=settings.get(SETTINGS_IMAGE_GEN_IMAGE_SIZE),  # 1024x1024, 1536x1024, etc.
        background=settings.get(SETTINGS_IMAGE_GEN_BACKGROUND),  # "auto", "transparent", "opaque"
        quality=settings.get(SETTINGS_IMAGE_GEN_OUTPUT_COMPRESSION),  # 0-100
        style="vivid",  # Default style
        **kwargs
    )
    return response
```

GPT-Image-1 supports several configuration options in VT.ai:

- **Image Size**: Control the dimensions of generated images
  - `1024x1024` (square - default)
  - `1536x1024` (landscape)
  - `1024x1536` (portrait)

- **Background Type**: Control transparency
  - `auto` - Let the model decide (default)
  - `transparent` - Create images with transparent backgrounds (for PNG format)
  - `opaque` - Force an opaque background

- **Output Format**: Select image format
  - `jpeg` - Good for photographs (default)
  - `png` - Best for images needing transparency
  - `webp` - Optimized for web use with good compression

- **Moderation Level**: Content filtering level
  - `auto` - Standard moderation (default)
  - `low` - Less restrictive moderation

- **Compression Quality**: For JPEG and WebP formats
  - Values from 0-100 (75 is default)
  - Higher values produce better quality but larger files

## Error Handling

VT.ai implements robust error handling for model calls:

- API rate limiting errors
- Authentication errors
- Model-specific errors
- Network errors

The main error handling is centralized in `vtai/utils/error_handlers.py`.

## Model Performance

### Streaming Responses

VT.ai supports streaming responses for a better user experience:

```python
# Example streaming call
async def stream_model_response(messages, model="o3-mini", **kwargs):
    from litellm import acompletion

    response_stream = await acompletion(
        model=model,
        messages=messages,
        stream=True,
        **kwargs
    )

    collected_content = ""
    async for chunk in response_stream:
        content = chunk.choices[0].delta.content
        if content:
            collected_content += content
            # Handle chunk processing

    return collected_content
```

### Caching

VT.ai implements caching for model responses to improve performance and reduce API costs:

- In-memory cache for short-term use
- Disk-based cache for persistent storage

## Adding New Model Providers

To add support for a new model provider:

1. Update the provider configuration in `llm_providers_config.py`
2. Add the appropriate API key handling
3. Test compatibility with the semantic router
4. Implement any specialized handling if needed

See the [Extending VT.ai](extending.md) guide for more details.

*This page is under construction. More detailed information about model integration will be added soon.*
