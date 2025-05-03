# VT.ai Features

This page provides detailed information about VT.ai's key features and how to use them effectively.

## Chat Modes

VT.ai offers different chat modes to suit your specific needs:

### Standard Chat

The standard chat mode provides access to all configured LLM providers with dynamic conversation routing:

- Automatic classification and routing of queries to appropriate models
- Support for text, image, and audio inputs
- Full access to all VT.ai features

To use standard chat:

1. Simply type your message in the input field
2. Press Enter to send
3. The system will automatically route your query to the most appropriate model

### Assistant Mode (Beta)

Assistant mode provides specialized capabilities for more complex tasks:

- Code interpreter for executing Python code
- File attachment support (PDF, CSV, images, etc.)
- Persistent conversation threads
- Function calling for external integrations
- Web search with intelligent summarization

To use assistant mode:

1. Switch to the Assistant profile in the dropdown menu
2. Upload files if needed
3. Type your queries as normal
4. View the step-by-step execution in the interface

## Specialized Features

### Web Search with Smart Summarization

VT.ai can search the web for information and intelligently summarize the results:

- Accumulate information from multiple search results
- Generate coherent, comprehensive summaries
- Cite sources with proper attribution
- Toggle between raw results and AI-synthesized summaries

To use web search with summarization:

1. Ask a question that might require current information
2. VT.ai will automatically route to the web search tool
3. Results will be summarized into a concise, readable answer
4. Sources will be listed with clickable links

You can control summarization behavior:

- Enable/disable summarization in the settings menu
- When enabled, multiple search results are synthesized into a unified response
- When disabled, search results are presented in a more raw format

Example queries:

- "What are the latest developments in quantum computing?"
- "Search for information about sustainable energy solutions"
- "Find recent news about Mars exploration"

### Thinking Mode

Thinking mode gives you access to step-by-step reasoning from the models, providing transparency into the AI's thought process:

- See the model's internal reasoning process
- Understand how the model arrived at its conclusion
- Great for learning, debugging, and complex problem-solving
- Helps with verification of facts and logical reasoning

#### How Thinking Mode Works

In the background, VT.ai uses special reasoning-enhanced models and prompt engineering to make the model's thought process explicit:

1. The query is sent to a reasoning-capable model with instructions to show its work
2. The model breaks down the problem into steps
3. Each step of reasoning is displayed in the interface
4. The final conclusion is presented after the reasoning steps

#### Using Thinking Mode

There are two ways to activate thinking mode:

1. **Manual Activation**: Add the `<think>` tag at the beginning of your message

   ```
   <think>What are the key factors that contributed to the Industrial Revolution?
   ```

2. **Automatic Activation**: Enable "Use Thinking Mode For Reasoning Models" in settings
   - When enabled, VT.ai will automatically use thinking mode for models in the reasoning-enhanced list
   - This includes models like DeepSeek Reasoner, DeepSeek R1 series, Qwen 2.5, Claude 3, and GPT-4o

#### Best Uses for Thinking Mode

Thinking mode is especially useful for:

- **Complex problem solving**: Mathematics, logic puzzles, step-by-step analysis
- **Fact verification**: See how the model reaches factual conclusions
- **Learning**: Understand reasoning processes for educational topics
- **Debugging**: Identify where reasoning might go wrong
- **Decision making**: Follow the model's decision process

Example queries that work well with thinking mode:

```
<think>Is it more environmentally friendly to use paper bags or plastic bags?

<think>Solve the quadratic equation: 2x² + 7x - 15 = 0

<think>What would happen to Earth's climate if the sun suddenly became 10% brighter?

<think>Analyze the following code and explain what it does:
def mystery_function(arr):
    result = []
    for i in range(len(arr)):
        if i % 2 == 0:
            result.append(arr[i] * 2)
        else:
            result.append(arr[i] + 3)
    return result
```

#### Models That Excel with Thinking Mode

While thinking mode works with all models, these models are specifically optimized for step-by-step reasoning:

- **DeepSeek Reasoner**: Specifically designed for transparent reasoning
- **DeepSeek R1 Series**: Enhanced reasoning capabilities across different model sizes
- **Qwen 2.5 Models**: Excellent structured reasoning abilities
- **Claude 3 Opus/Sonnet**: Strong logical reasoning with clear explanations
- **GPT-4o**: Advanced reasoning with multimodal capabilities

### Image Analysis

VT.ai can analyze and interpret images:

- Upload images directly from your device
- Provide URLs to online images
- Get detailed descriptions and analysis
- Extract text with optical character recognition (OCR)
- Identify objects, scenes, and visual elements

To analyze an image:

1. Click the upload button or paste an image URL
2. Ask a question about the image
3. The system will analyze the image and respond to your query

Example queries:

- "What's in this image?"
- "Can you describe this diagram?"
- "What text appears in this screenshot?"
- "Identify the objects in this picture"
- "What emotion does the person in this image appear to be feeling?"

### Image Generation

Generate images based on text descriptions:

- **DALL-E 3**: Create custom images from detailed prompts
- **GPT-Image-1**: Advanced image generation with extensive customization options
  - Transparent backgrounds for logos and graphics
  - Multiple output formats (PNG, JPEG, WEBP)
  - Customizable dimensions (square, landscape, portrait)
  - Variable quality and compression settings
  - HD options for higher quality outputs
  - Moderation controls

To generate an image:

1. Type a prompt like "Generate an image of a futuristic city with flying cars"
2. The system will recognize the image generation intent
3. The appropriate image generation model will create and display the image based on your description

#### Advanced Image Generation Settings

For advanced GPT-Image-1 options, you can configure:

- **Image size**: Set with `VT_SETTINGS_IMAGE_GEN_IMAGE_SIZE`
  - Options: "1024x1024" (square), "1792x1024" (landscape), "1024x1792" (portrait), "1536x1536" (large square)

- **Quality**: Set with `VT_SETTINGS_IMAGE_GEN_IMAGE_QUALITY`
  - Options: "standard" (faster), "hd" (higher quality)

- **Background**: Set with `VT_SETTINGS_IMAGE_GEN_BACKGROUND`
  - Options: "auto" (context-dependent), "transparent" (for PNG format)

- **Output format**: Set with `VT_SETTINGS_IMAGE_GEN_OUTPUT_FORMAT`
  - Options: "png" (lossless, supports transparency), "jpeg" (smaller file size), "webp" (best compression)

- **Compression**: Set with `VT_SETTINGS_IMAGE_GEN_OUTPUT_COMPRESSION`
  - Range: 0-100, where 100 is maximum quality (webp/jpeg only)

#### GPT-Image-1 Prompt Guide

To get the best results with GPT-Image-1, consider these prompting strategies:

- **Be specific and detailed**: Describe subjects, setting, lighting, style, mood
- **Specify artistic style**: Photorealistic, cartoon, oil painting, watercolor, etc.
- **Include perspective information**: Close-up, aerial view, isometric, etc.
- **Mention lighting conditions**: Natural light, studio lighting, dramatic shadows
- **Reference time period or era**: Victorian, futuristic, 1980s, etc.

Example of a detailed prompt:

```
Generate an image of a serene Japanese garden at sunset, with a small wooden bridge crossing a koi pond. Cherry blossom trees frame the scene, with soft pink petals falling onto the water's surface. The lighting is warm and golden, creating long shadows. Style: watercolor painting with fine details.
```

### Voice Interaction

VT.ai supports comprehensive voice-based interaction:

- **Speech-to-Text**: Real-time voice transcription using OpenAI's Whisper model
  - Smart silence detection for natural conversation flow
  - High-accuracy transcription across multiple languages
  - Seamless integration with conversation routing

- **Text-to-Speech**: Listen to AI responses with natural-sounding voices
  - Multiple voice options (alloy, echo, fable, onyx, nova, shimmer)
  - Toggle TTS on/off in settings
  - High-quality voice synthesis using OpenAI's Audio API
  - Speak response action appears on all messages

- **Audio Understanding**: Analyze and understand audio content
  - Upload audio files for detailed analysis
  - Get both transcription and contextual understanding
  - Support for various audio formats (MP3, WAV, M4A, etc.)

To use voice features:

1. Enable TTS in the settings menu
2. Select your preferred voice model
3. Each response will include a speech button to listen to the content
4. For voice input, click the microphone icon and speak your query

For the best voice interaction experience:

- Use a good quality microphone in a quiet environment
- Speak clearly and at a moderate pace
- Allow a brief pause after speaking to trigger automatic detection
- Choose a voice model that matches your preference for response playback

## Model Selection

VT.ai supports a wide range of models:

- **OpenAI**: GPT-o1, GPT-o1 Mini, GPT-o1 Pro, GPT-o3 Mini, GPT-4.5 Preview, GPT-4o
- **Anthropic**: Claude 3.5/3.7 (Sonnet, Haiku, Opus)
- **Google**: Gemini 1.5 Pro/Flash, Gemini 2.0, Gemini 2.5 Pro/Flash
- **Vision Models**: GPT-4o, Gemini 1.5 Pro/Flash, Gemini 2.5 Pro/Flash, Claude 3 models, Llama 3.2 Vision
- **TTS Models**: GPT-4o mini TTS, TTS-1, TTS-1-HD
- **Local Models**: Llama3, Mistral, DeepSeek R1, Qwen2.5-coder (via Ollama)

You can select models in several ways:

1. Use the model selector in the settings menu
2. Specify a model at startup with `vtai --model model-name`
3. Let the semantic router automatically select the best model for your query

## Configuration Options

VT.ai offers various configuration options accessible through the settings menu:

- **Temperature**: Control randomness in responses (0.0-2.0)
- **Top P**: Adjust response diversity (0.0-1.0)
- **Image Generation Settings**: Style, quality, format, and dimension options
- **TTS Settings**: Voice models and quality options
- **Routing Options**: Enable/disable dynamic conversation routing
- **Thinking Mode**: Enable/disable automatic thinking mode for reasoning models
- **Web Search**: Configure search result display and summarization

For more details on configuration, see the [Configuration](configuration.md) page.
