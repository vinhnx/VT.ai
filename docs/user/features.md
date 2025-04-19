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

Thinking mode gives you access to step-by-step reasoning from the models:

- See the model's internal reasoning process
- Understand how the model arrived at its conclusion
- Great for learning, debugging, and complex problem-solving

To use thinking mode:

1. Add the `<think>` tag at the beginning of your message
2. The model will show its reasoning steps before providing the final answer
3. You can see both the thinking process and the final response

Example:

```
<think>What are the key factors that contributed to the Industrial Revolution?
```

### Image Analysis

VT.ai can analyze and interpret images:

- Upload images directly from your device
- Provide URLs to online images
- Get detailed descriptions and analysis

To analyze an image:

1. Click the upload button or paste an image URL
2. Ask a question about the image
3. The system will analyze the image and respond to your query

Example queries:

- "What's in this image?"
- "Can you describe this diagram?"
- "What text appears in this screenshot?"

### Image Generation

Generate images based on text descriptions using DALL-E 3:

- Create custom images from detailed prompts
- Control style and quality parameters
- Visualize concepts and ideas

To generate an image:

1. Type a prompt like "Generate an image of a futuristic city with flying cars"
2. The system will recognize the image generation intent
3. DALL-E 3 will create and display the image based on your description

### Voice Interaction

VT.ai supports voice-based interaction:

- Text-to-speech for hearing responses
- Multiple voice models to choose from
- Natural speech synthesis

To use voice features:

1. Enable TTS in the settings menu
2. Select your preferred voice model
3. Each response will include a speech button to listen to the content

## Model Selection

VT.ai supports a wide range of models:

- **OpenAI**: GPT-o1, GPT-o3 Mini, GPT-4o
- **Anthropic**: Claude 3.5/3.7 (Sonnet, Opus)
- **Google**: Gemini 2.0/2.5
- **Vision Models**: GPT-4o, Gemini 1.5 Pro/Flash, Llama3.2 Vision
- **TTS Models**: GPT-4o mini TTS, TTS-1, TTS-1-HD
- **Local Models**: Llama3, Mistral, DeepSeek R1 (via Ollama)

You can select models in several ways:

1. Use the model selector in the settings menu
2. Specify a model at startup with `vtai --model model-name`
3. Let the semantic router automatically select the best model for your query

## Configuration Options

VT.ai offers various configuration options accessible through the settings menu:

- **Temperature**: Control randomness in responses
- **Top P**: Adjust response diversity
- **Image Generation Settings**: Style and quality options
- **TTS Settings**: Voice models and quality options
- **Routing Options**: Enable/disable dynamic conversation routing

For more details on configuration, see the [Configuration](configuration.md) page.
