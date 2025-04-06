
<p align="center">
  <h1 align="center">VT.ai</h1>

  <img src="./public/screenshot.jpg" alt="screenshot" />

  Multimodal AI chat application with semantic-based conversation routing

  [![Open in GitHub Codespaces](https://img.shields.io/badge/Open%20in-Codespaces-blue?logo=github)](https://codespaces.new/vinhnx/VT.ai)
  [![Twitter Follow](https://img.shields.io/twitter/follow/vinhnx?style=social)](https://x.com/vinhnx)
  [![Twitter Follow](https://img.shields.io/twitter/follow/vtdotai?style=social)](https://x.com/vtdotai)
</p>

## Features

### Multi-Provider AI Integration
**Supported AI Model Providers**:
- OpenAI (GPT-o1, GPT-o3, GPT-4.5, GPT-4o)
- Anthropic (Claude 3.5, Claude 3.7 Sonnet)
- Google (Gemini 1.5/2.0/2.5 Pro/Flash series)
- DeepSeek (DeepSeek R1 and V3 series)
- Meta (Llama 3 & 4, including Maverick & Scout)
- Cohere (Command, Command-R, Command-R-Plus)
- Local Models via Ollama (Llama3, Phi-3, Mistral, DeepSeek R1)
- Groq (Llama 3 70B, Mixtral 8x7B)
- OpenRouter (unified access to multiple providers)

**Core Capabilities**:
- Semantic-based routing using embedding-based classification
- Multi-modal interactions across text, image, and audio
- Vision analysis for images and URLs
- Image generation with DALL-E 3
- Text-to-Speech with multiple voice models
- Assistant framework with code interpreter
- Thinking mode with transparent reasoning steps
- Real-time streaming responses
- Provider-agnostic model switching

## Architecture

VT.ai implements a modular architecture centered on a semantic routing system that directs user queries to specialized handlers:

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
   - Vector-based query classification system using FastEmbed embeddings
   - Routes user queries to specialized handlers using the `SemanticRouterType` enum
   - Supports five distinct routing categories:
     - Text processing (summaries, translations, analysis)
     - Image generation (DALL-E prompt crafting)
     - Vision analysis (image interpretation)
     - Casual conversation (social interactions)
     - Curious inquiries (informational requests)

2. **Model Management** (`/src/utils/llm_settings_config.py`)
   - Unified API abstractions through LiteLLM
   - Provider-agnostic model switching with dynamic parameters
   - Centralized configuration for model settings and routing

3. **Conversation Handling** (`/src/utils/conversation_handlers.py`)
   - Streaming response processing with backpressure handling
   - Multi-modal content parsing and rendering
   - Thinking mode implementation showing reasoning steps
   - Error handling with graceful fallbacks

4. **Assistant Framework** (`/src/assistants/mino/`)
   - Tool use capabilities with function calling
   - Code interpreter integration for computation
   - File attachment processing with multiple formats
   - Assistant state management

5. **Media Processing** (`/src/utils/media_processors.py`)
   - Image generation pipeline for DALL-E 3
   - Vision analysis with cross-provider model support
   - Audio transcription and Text-to-Speech integration

## Setup Guide

### Prerequisites
- Python 3.11+ (specified in `.python-version`)
- [uv](https://github.com/astral-sh/uv) for dependency management
- [Ollama](https://github.com/ollama/ollama) (optional, for local models)

### Installation

```bash
# Clone repository
git clone https://github.com/vinhnx/VT.ai.git
cd VT.ai

# Setup environment using uv
uv venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies using uv
uv pip install -e .         # Install main dependencies
uv pip install -e ".[dev]"  # Optional: Install development dependencies

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Edit the `.env` file with your API keys for the models you intend to use:

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
.venv\Scripts\activate     # Windows

# Launch the interface
chainlit run src/app.py -w
```

The `-w` flag enables auto-reloading during development.

### Dependency Management

The project uses uv for faster and more reliable dependency management:

```bash
# Add a new dependency
uv pip install package-name

# Update a dependency
uv pip install --upgrade package-name

# Export dependencies to requirements.txt
uv pip freeze > requirements.txt

# Install from requirements.txt
uv pip install -r requirements.txt
```

### Customizing the Semantic Router

To customize the semantic router for specific use cases:

```bash
python src/router/trainer.py
```

This utility updates the `layers.json` file with new routing rules and requires an OpenAI API key to generate embeddings.

## Usage Guide

### Interface Controls

| Shortcut | Action                      |
|----------|----------------------------- |
| Ctrl+/   | Switch model provider       |
| Ctrl+,   | Open settings panel         |
| Ctrl+L   | Clear conversation history  |

### Chat Modes

1. **Standard Chat**
   - Access to all configured LLM providers
   - Dynamic conversation routing based on query classification
   - Support for text, image, and audio inputs
   - Advanced thinking mode with reasoning trace (use "<think>" tag)

2. **Assistant Mode (Beta)**
   - Code interpreter for computations and data analysis
   - File attachment support (PDF/CSV/Images)
   - Persistent conversation threads
   - Function calling for external integrations

### Specialized Features

- **Image Generation**: Generate images through prompts ("Generate an image of...")
- **Image Analysis**: Upload or provide URL for image interpretation
- **Text Processing**: Request summaries, translations, or content transformation
- **Voice Interaction**: Use speech recognition for input and TTS for responses
- **Thinking Mode**: Access step-by-step reasoning from the models

## Supported Models

| Category       | Models                                                     |
|----------------|-----------------------------------------------------------|
| **Chat**       | GPT-o1, GPT-o3 Mini, GPT-4o, Claude 3.5/3.7, Gemini 2.0/2.5 |
| **Vision**     | GPT-4o, Gemini 1.5 Pro/Flash, Llama3.2 Vision             |
| **Image Gen**  | DALL-E 3                                                   |
| **TTS**        | GPT-4o mini TTS, TTS-1, TTS-1-HD                          |
| **Local**      | Llama3, Mistral, DeepSeek R1 (1.5B to 70B)                |

## Development

### Environment Setup

```bash
# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install development dependencies
uv pip install -e ".[dev]"

# Format code
black .

# Run linting
flake8 src/
isort src/

# Run tests
pytest
```

### Dependency Management

```bash
# Add a project dependency
uv pip install package-name
# Update pyproject.toml manually after confirming compatibility

# Add a development dependency
uv pip install --dev package-name
# Update pyproject.toml's project.optional-dependencies.dev section

# Sync all project dependencies after pulling updates
uv pip install -e .
uv pip install -e ".[dev]"
```

### Contribution Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-capability`)
3. Add type hints for new functions
4. Update documentation to reflect changes
5. Submit a Pull Request with comprehensive description

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgements

- [Chainlit](https://chainlit.io) - Chat interface framework
- [LiteLLM](https://docs.litellm.ai) - Model abstraction layer
- [SemanticRouter](https://github.com/aurelio-labs/semantic-router) - Intent classification
- [FastEmbed](https://github.com/qdrant/fastembed) - Embedding models for routing

## Contact

I'm [@vinhnx](https://github.com/vinhnx) on the internet.

Thank you, and have a great day!
