<h1 align="center">VT</h1>

<p align="center">
  <img src="./public/screenshot.jpg" alt="VT.ai screenshot">
  <p align="center">Minimal Multimodal AI Chat App with Dynamic Routing</p>
</p>

<p align="center">
<a href="https://github.com/sponsors/vinhnx"><img alt="Github Sponsors" src="https://img.shields.io/badge/GitHub%20Sponsors-30363D?&logo=GitHub-Sponsors&logoColor=EA4AAA"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/vtai/"><img alt="PyPI" src="https://img.shields.io/pypi/v/vtai?logo=pypi&logoColor=fff"></a>
  <a href="https://pypi.org/project/vtai/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/vtai?logo=python&logoColor=white"></a>
  <a href="https://huggingface.co/vinhnx90"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000"></a>
  <a href="https://codespaces.new/vinhnx/VT.ai"><img alt="Open in GitHub Codespaces" src="https://img.shields.io/badge/Open%20in-Codespaces-blue?logo=github"/></a>
  <a href="https://vinhnx.github.io/VT.ai"><img alt="Documentation" src="https://img.shields.io/badge/Documentation-526CFE?logo=materialformkdocs&logoColor=fff"></a>
  <a href="https://github.com/vinhnx/VT.ai/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

<p align="center">
   <a href="#"><img alt="Google Gemini" src="https://img.shields.io/badge/Google%20Gemini-886FBF?logo=googlegemini&logoColor=fff"></a>
  <a href="#"><img alt="Claude" src="https://img.shields.io/badge/Claude-D97757?logo=claude&logoColor=fff"></a>
  <a href="#"><img alt="ChatGPT" src="https://img.shields.io/badge/ChatGPT-74aa9c?logo=openai&logoColor=white"></a>
  <a href="#"><img alt="Deepseek" src="https://custom-icon-badges.demolab.com/badge/Deepseek-4D6BFF?logo=deepseek&logoColor=fff"></a>
</p>

## VT.ai

VT.ai is a multimodal AI chat application designed to simplify interaction with different AI models through a unified interface. It employs vector-based semantic routing to direct queries to the most suitable model, eliminating the need to switch between multiple applications and interfaces.

**[Full documentation available here](https://vinhnx.github.io/VT.ai/)**

## Key Features

- **Multi-Provider Integration**: Unified access to models from OpenAI (o1/o3/4o), Anthropic (Claude), Google (Gemini), DeepSeek, Llama, Cohere, and local models via Ollama
- **Semantic Routing System**: Vector-based classification automatically routes queries to appropriate models using FastEmbed embeddings, removing the need for manual model selection
- **Multimodal Capabilities**: Comprehensive support for text, image, and audio inputs with advanced vision analysis
- **Image Generation**: GPT-Image-1 integration with support for transparent backgrounds, multiple formats, and customizable quality parameters
- **Web Search Integration**: Real-time information retrieval with source attribution via Tavily API
- **Voice Processing**: Advanced speech-to-text and text-to-speech functionality with configurable voice options and silence detection
- **Reasoning Visualization**: Step-by-step model reasoning visualization with the `<think>` tag for transparent AI decision processes

## Installation & Setup

Multiple installation methods are available depending on requirements:

```bash
# Standard PyPI installation
uv pip install vtai

# Zero-installation experience with uvx
export OPENAI_API_KEY='your-key-here'
uvx vtai

# Development installation
git clone https://github.com/vinhnx/VT.ai.git
cd VT.ai
uv venv
source .venv/bin/activate  # Linux/Mac
uv pip install -e ".[dev]"  # Install with development dependencies
```

### API Key Configuration

Configure API keys to enable specific model capabilities:

```bash
# Command-line configuration
vtai --api-key openai=sk-your-key-here

# Environment variable configuration
export OPENAI_API_KEY='sk-your-key-here'  # For OpenAI models
export ANTHROPIC_API_KEY='sk-ant-your-key-here'  # For Claude models
export GEMINI_API_KEY='your-key-here'  # For Gemini models
```

API keys are securely stored in `~/.config/vtai/.env` for future use.

## Usage Guide

### Programmatic Usage

```python
from vtai.app import run_app

# Basic usage with default settings
run_app()

# Advanced configuration
run_app(
    models=["gpt-4o", "claude-3-5-sonnet"],
    enable_web_search=True,
    enable_voice=True,
    enable_thinking=True
)
```

### Interface Usage

The application provides a clean, intuitive interface with the following capabilities:

1. **Dynamic Conversations**: The semantic router automatically selects the most appropriate model for each query
2. **Image Generation**: Create images using prompts like "generate an image of..." or "draw a..."
3. **Visual Analysis**: Upload or provide URLs to analyze visual content
4. **Reasoning Visualization**: Add `<think>` to prompts to observe step-by-step reasoning
5. **Voice Interaction**: Use the microphone feature for speech input and text-to-speech output

Detailed usage instructions are available in the [Getting Started Guide](https://vinhnx.github.io/VT.ai/user/getting-started/).

## Documentation

The documentation is organized into sections designed for different user needs:

- **[User Guide](https://vinhnx.github.io/VT.ai/user/getting-started/)**: Installation, configuration, and feature documentation
- **[Developer Guide](https://vinhnx.github.io/VT.ai/developer/architecture/)**: Architecture details, extension points, and implementation information
- **[API Reference](https://vinhnx.github.io/VT.ai/api/)**: Comprehensive API documentation for programmatic usage

## Implementation Options

VT.ai offers two distinct implementations:

- **Python Implementation**: Full-featured reference implementation with complete support for all capabilities
- **Rust Implementation**: High-performance alternative with optimized memory usage and native compiled speed

The [implementation documentation](https://vinhnx.github.io/VT.ai/user/getting-started/#implementation-options) provides a detailed comparison of both options.

## Supported Models

| Category       | Models                                                |
|----------------|----------------------------------------------------- |
| **Chat**       | GPT-o1, GPT-o3 Mini, GPT-4o, Claude 3.5/3.7, Gemini 2.0/2.5  |
| **Vision**     | GPT-4o, Gemini 1.5 Pro/Flash, Claude 3, Llama3.2 Vision    |
| **Image Gen**  | GPT-Image-1 with custom parameters                   |
| **TTS**        | GPT-4o mini TTS, TTS-1, TTS-1-HD                     |
| **Local**      | Llama3, Mistral, DeepSeek R1 (1.5B to 70B via Ollama) |

The [Models Documentation](https://vinhnx.github.io/VT.ai/user/models/) provides detailed information about model-specific capabilities and configuration options.

## Technical Architecture

VT.ai leverages several open-source projects to deliver its functionality:

- **[Chainlit](https://chainlit.io)**: Modern chat interface framework
- **[LiteLLM](https://docs.litellm.ai)**: Unified model abstraction layer
- **[SemanticRouter](https://github.com/aurelio-labs/semantic-router)**: Intent classification system
- **[FastEmbed](https://github.com/qdrant/fastembed)**: Efficient embedding generation
- **[Tavily](https://tavily.com)**: Web search capabilities

The application architecture follows a clean, modular design:

- **Entry Point**: `vtai/app.py` - Main application logic
- **Routing Layer**: `vtai/router/` - Semantic classification system
- **Assistants**: `vtai/assistants/` - Specialized handlers for different query types
- **Tools**: `vtai/tools/` - Web search, file operations, and other integrations

## Contributing

Contributions to VT.ai are welcome. The project accepts various types of contributions:

- **Bug Reports**: Submit detailed GitHub issues for any bugs encountered
- **Feature Requests**: Propose new functionality through GitHub issues
- **Pull Requests**: Submit code improvements and bug fixes
- **Documentation**: Enhance documentation or add examples
- **Feedback**: Share user experiences to help improve the project

Development setup:

```bash
# Clone the repository
git clone https://github.com/vinhnx/VT.ai.git
cd VT.ai

# Set up development environment
uv venv
source .venv/bin/activate  # Linux/Mac
uv pip install -e ".[dev]"

# Run tests
pytest
```

## Testing and Quality

Quality is maintained through comprehensive testing:

```bash
# Run the test suite
pytest

# Run with coverage reporting
pytest --cov=vtai

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

## License

VT.ai is available under the MIT License - See [LICENSE](LICENSE) for details.

## Contact

Contact [@vinhnx](https://github.com/vinhnx) on GitHub with questions or feedback about VT.ai.
