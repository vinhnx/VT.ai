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

## Installation

VT.ai can be installed and run in multiple ways depending on your needs:

### Quick Install from PyPI

```bash
# Install VT.ai from PyPI
pip install vtai

# Run the application
vtai
```

### Install with uv

```bash
# If you need to install uv first
python -m pip install uv

# Install VT.ai with uv
uv tool install --force --python python3.11 vtai@latest
```

This will install uv using your existing Python version and use it to install VT.ai. If needed, uv will automatically install a separate Python 3.11 to use with VT.ai.

### Install with pipx

```bash
# If you need to install pipx first
python -m pip install pipx

# Install VT.ai with pipx
pipx install vtai
```

You can use pipx to install VT.ai with Python versions 3.9-3.12.

### Development Install (from source)

```bash
# Clone repository
git clone https://github.com/vinhnx/VT.ai.git
cd VT.ai

# Setup environment using uv
uv venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e .
```

### Platform-Specific Instructions

#### macOS

```bash
# Install Python 3.11+ if needed
brew install python@3.11

# Install VT.ai
pip3 install vtai

# Run
vtai
```

#### Linux

```bash
# Ensure Python 3.11+ is installed
sudo apt install python3.11 python3.11-venv  # Ubuntu/Debian
# or
sudo dnf install python3.11  # Fedora

# Install VT.ai
pip3 install vtai

# Run
vtai
```

#### Windows

```powershell
# Install VT.ai
pip install vtai

# Run
vtai
```

### Command-Line Options

```bash
# Show help and options
vtai --help

# Run with hot-reload for development
vtai -w
```

## Setup Guide

### Prerequisites
- Python 3.11+ (specified in `.python-version`)
- [uv](https://github.com/astral-sh/uv) for dependency management (recommended)
- [Ollama](https://github.com/ollama/ollama) (optional, for local models)

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

### Starting from Source

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

## Troubleshooting

### ModuleNotFoundError: No module named 'vtai'

If you encounter this error when running `vtai -w`:

```
ModuleNotFoundError: No module named 'vtai'
```

This usually happens when you've installed VT.ai using `uv tool install` or a similar method that puts the package in an isolated environment, but you're trying to run it from within the source directory.

#### Solutions:

1. **Use the development install method instead:**

   ```bash
   # In the VT.ai repository root
   uv venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   
   # Install in development mode
   uv pip install -e .
   
   # Run directly with chainlit
   chainlit run vtai/app.py -w
   ```

2. **Run the installed vtai command outside of the source directory:**

   ```bash
   # Move to a different directory
   cd ..
   
   # Then run vtai
   vtai -w
   ```

3. **Update your PYTHONPATH:**

   ```bash
   # Linux/Mac
   export PYTHONPATH=/path/to/VT.ai:$PYTHONPATH
   
   # Windows
   set PYTHONPATH=C:\path\to\VT.ai;%PYTHONPATH%
   ```

### For other issues

If you're experiencing other problems, please check:

1. That all required API keys are set in your `.env` file
2. Python version is 3.11 or higher
3. All dependencies are properly installed

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
