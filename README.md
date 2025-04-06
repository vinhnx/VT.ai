<p align="center">
    <img src="./vtai/resources/vt.jpg" alt="VT.ai Logo" width="300">
</p>

<h1 align="center">
VT - Minimal Multimodal AI Chat App with Dynamic Routing
</h1>

<p align="center">
  <img src="./public/screenshot.jpg" alt="VT.ai screenshot">
</p>

<p align="center">
  <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/vtai">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/vinhnx/vt.ai">
  <a href="https://codespaces.new/vinhnx/VT.ai"><img alt="Open in GitHub Codespaces" src="https://img.shields.io/badge/Open%20in-Codespaces-blue?logo=github"/></a>
  <a href="https://x.com/vinhnx"><img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/vinhnx?style=social"/></a>
  <a href="https://x.com/vtdotai"><img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/vtdotai?style=social"/></a>
</p>

## Features

### [Multi-Provider AI Integration](#multi-provider-ai-integration)

Supports OpenAI (o1, o3, 4o), Anthropic (Claude 3.5, 3.7), Google (Gemini series), DeepSeek, Meta (Llama), Cohere, local models via Ollama, and more.

### [Semantic-Based Routing](#semantic-router)

Smart routing system that automatically directs queries to specialized handlers based on vector-based classification.

### [Multimodal Capabilities](#core-capabilities)

Support for text, image, and audio inputs with vision analysis for images and URLs, plus DALL-E 3 image generation.

### [Voice Interaction](#specialized-features)

Use speech recognition for input and text-to-speech for responses with multiple voice models.


### [Assistant Framework](#assistant-framework)

Code interpreter for computations and data analysis with function calling for external integrations.

### [Thinking Mode](#chat-modes)

Access step-by-step reasoning from the models with transparent thinking processes.

## Getting Started

```bash
# Install VT.ai from PyPI
pip install vtai

# Run the application
vtai

# With additional arguments
vtai --help  # Show command-line options
vtai -w      # Run with auto-reload for development
```

See the [Setup Guide](#setup-guide) for more detailed instructions on installation and configuration.

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

## Key Components

### Semantic Router

- Vector-based query classification using FastEmbed embeddings
- Routes user queries to specialized handlers
- Supports five distinct routing categories:
  - Text processing (summaries, translations, analysis)
  - Image generation (DALL-E prompt crafting)
  - Vision analysis (image interpretation)
  - Casual conversation (social interactions)
  - Curious inquiries (informational requests)

### Model Management

- Unified API abstractions through LiteLLM
- Provider-agnostic model switching with dynamic parameters
- Centralized configuration for model settings and routing

### Conversation Handling

- Streaming response processing with backpressure handling
- Multi-modal content parsing and rendering
- Thinking mode implementation showing reasoning steps
- Error handling with graceful fallbacks

### Assistant Framework

- Tool use capabilities with function calling
- Code interpreter integration for computation
- File attachment processing with multiple formats
- Assistant state management

### Media Processing

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
chainlit run vtai/app.py -w
```

The `-w` flag enables auto-reloading during development.

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
flake8 vtai/
isort vtai/

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

## More Information

### Documentation
- [Setup Guide](#setup-guide)
- [Usage Guide](#usage-guide)
- [Supported Models](#supported-models)
- [Architecture](#architecture)
- [Key Components](#key-components)

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
