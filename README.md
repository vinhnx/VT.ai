---
<p align="center">
  <img src="./public/logo_dark.png" height="200" alt="VT.ai Logo" />
  <h1 align="center">VT.ai</h1>
  <p align="center">Minimal multimodal AI chat app with dynamic conversation routing</p>

  [![Open in GitHub Codespaces](https://img.shields.io/badge/Open%20in-Codespaces-blue?logo=github)](https://codespaces.new/vinhnx/VT.ai)
  <!--[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)-->
  [![Twitter Follow](https://img.shields.io/twitter/follow/vtdotai?style=social)](https://twitter.com/vtdotai)
</p>

## 🚀 Features

### Multi-Provider AI Orchestration
✅ **Supported AI Model Providers**:
- OpenAI (GPT-o1, GPT-o3, GPT-4.5, GPT-4o)
- Anthropic (Claude 3.5, Claude 3.7 Sonnet)
- Google (Gemini 1.5/2.0/2.5 Pro/Flash series)
- DeepSeek (DeepSeek R1 and V3 series)
- Meta (Llama 3 & 4, including Maverick & Scout)
- Cohere (Command, Command-R, Command-R-Plus)
- Local Models via Ollama (Llama3, Phi-3, Mistral, DeepSeek R1)
- Groq (Llama 3 70B, Mixtral 8x7B)
- OpenRouter (for accessing multiple providers with one key)

✨ **Core Capabilities**:
- Dynamic conversation routing with semantic understanding
- Multi-modal interactions (Text/Image/Audio)
- Vision analysis for images and URLs
- Image generation with DALL-E 3
- Text-to-Speech with latest voice models
- Assistant framework with code interpreter
- Thinking mode with visible reasoning process
- Real-time response streaming
- Cross-provider model switching

## 🏗️ Architecture

VT.ai uses a modular architecture built around intelligent request routing and multi-provider model support:

```
┌────────────────┐     ┌─────────────────┐     ┌───────────────────────┐
│  User Request  │────▶│ Semantic Router │────▶│ Route-Specific Handler │
└────────────────┘     └─────────────────┘     └───────────────────────┘
                             │                            │
                             ▼                            ▼
                     ┌──────────────┐           ┌──────────────────┐
                     │ Model Select │◀─────────▶│ Provider API     │
                     └──────────────┘           │ (OpenAI,Gemini,  │
                             │                  │  Anthropic,etc.) │
                             ▼                  └──────────────────┘
                     ┌──────────────┐
                     │  Response    │
                     │  Processing  │
                     └──────────────┘
```

### Key Components

1. **Semantic Router** (`/src/router/`)
   - Uses embeddings to classify user queries into intent categories
   - Routes requests to specialized handlers based on content type
   - Supports text processing, image generation, vision analysis, casual conversation, and curious inquiry

2. **Model Management** (`/src/utils/llm_settings_config.py`)
   - Unified API abstractions via LiteLLM
   - Provider-agnostic model switching
   - Configurable model parameters with dynamic loading

3. **Conversation Handling** (`/src/utils/conversation_handlers.py`)
   - Stream-based response processing
   - Multi-modal content handling
   - Thinking mode with visible reasoning process
   - Enhanced error handling and timeouts

4. **Assistant Framework** (`/src/assistants/`)
   - Tool-using capabilities
   - Code interpreter integration
   - File attachment processing

5. **Media Processing** (`/src/utils/media_processors.py`)
   - Image generation with DALL-E 3
   - Vision analysis with multiple model options
   - Audio transcription and Text-to-Speech

## 📦 Quick Start

### Prerequisites
- Python 3.11+ (specified in `.python-version`)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- [Ollama](https://github.com/ollama/ollama/blob/main/README.md#quickstart) (optional, for local models)

### Installation

```bash
# Clone repository
git clone https://github.com/vinhnx/VT.ai.git
cd VT.ai

# Setup environment
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Edit the `.env` file with your API keys based on which models you want to use:

```ini
# Required for basic functionality (at least one of these)
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key

# Optional providers (enable as needed)
ANTHROPIC_API_KEY=your-anthropic-key
COHERE_API_KEY=your-cohere-key
GROQ_API_KEY=your-groq-key
OPENROUTER_API_KEY=your-openrouter-key
MISTRAL_API_KEY=your-mistral-key
HUGGINGFACE_API_KEY=your-huggingface-key

# For local models
OLLAMA_HOST=http://localhost:11434
```

### Starting the Application

```bash
# Activate the virtual environment (if not already active)
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows

# Launch the interface
chainlit run src/app.py -w
```

The `-w` flag enables auto-reloading for development.

### (Optional) Train the Semantic Router

To customize the semantic router for your specific needs:

```bash
python src/router/trainer.py
```

This requires an OpenAI API key and will update the `layers.json` file with new routing rules.

## 💡 Usage Guide

### Chat Interface Controls

| Shortcut | Action                      |
|----------|----------------------------- |
| Ctrl+/   | Switch model provider       |
| Ctrl+,   | Open settings panel         |
| Ctrl+L   | Clear conversation history  |

### Available Chat Modes

1. **Standard Chat**
   - Interact with any configured LLM
   - Dynamic conversation routing based on query type
   - Support for text, images, and audio input
   - Thinking mode with visible reasoning process (use "<think>" tag)

2. **Assistant Mode (Beta)**
   - Code interpreter for complex calculations
   - File attachments (PDF/CSV/Images)
   - Persistent conversation threads
   - Function calling capabilities

### Task-Specific Features

- **Image Generation**: Ask for an image to be generated (e.g., "Generate an image of a mountain landscape")
- **Image Analysis**: Upload or provide a URL to an image for the AI to analyze
- **Text Processing**: Request summaries, translations, or other text transformations
- **Voice Input**: Use speech recognition for hands-free interaction
- **TTS Output**: Hear responses spoken aloud with configurable voices

## 🌐 Supported Models

| Category       | Models                                                     |
|----------------|-----------------------------------------------------------|
| **Chat**       | GPT-o1, GPT-o3 Mini, GPT-4o, Claude 3.5/3.7, Gemini 2.0/2.5 |
| **Vision**     | GPT-4o, Gemini 1.5 Pro/Flash, Llama3.2 Vision             |
| **Image Gen**  | DALL-E 3                                                   |
| **TTS**        | GPT-4o mini TTS, TTS-1, TTS-1-HD                          |
| **Local**      | Llama3, Mistral, DeepSeek R1 (1.5B to 70B)                |

## 🤝 Contributing

### Development Setup

```bash
# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install development tools
uv pip install pytest black

# Format code
black .
```

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add type hints for new functions
4. Update documentation
5. Open a Pull Request

## 📄 License

MIT License - See [LICENSE](LICENSE) for full text.

## 🌟 Acknowledgements

- [Chainlit](https://chainlit.io) - Chat interface framework
- [LiteLLM](https://docs.litellm.ai) - Model abstraction layer
- [SemanticRouter](https://github.com/aurelio-labs/semantic-router) - Intent classification
- [FastEmbed](https://github.com/qdrant/fastembed) - Embedding models for routing
