---
<p align="center">
  <img src="./public/logo_dark.png" height="200" alt="VT.ai Logo" />
</p>

<h1 align="center">VT.ai</h1>

<p align="center">
  <em>Multimodal AI Chat Application with Intelligent Conversation Routing</em>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/vinhnx/VT.ai/stargazers"><img src="https://img.shields.io/github/stars/vinhnx/VT.ai?style=social" alt="GitHub Stars"></a>
  <a href="https://twitter.com/vtdotai"><img src="https://img.shields.io/twitter/follow/vtdotai?style=social" alt="VT.ai Twitter"></a>
  <a href="https://twitter.com/vinhnx"><img src="https://img.shields.io/twitter/follow/vinhnx?style=social" alt="Creator Twitter"></a>
</p>

## 🌟 Overview

VT.ai is a sophisticated chat interface that integrates multiple AI providers and local models. Designed for developers and AI enthusiasts, it offers:

- Unified access to leading cloud AI APIs and local models
- Advanced conversation routing using semantic analysis
- Multi-modal interactions (text, images, audio)
- Customizable AI assistant capabilities

### Architectural Overview

<p align="center">
  <img src="./public/vtai_diagram.png" width="800" alt="System Architecture Diagram" />
</p>

*Diagram created using [gitdiagram](https://github.com/aurelio-labs/gitdiagram)*

## 🚀 Key Features

### Multi-modal Capabilities
- **Text & Image Processing**
  - Vision model integration (GPT-4o, Gemini 1.5 Pro, Llama 3.2 Vision)
  - Image generation with DALL-E 3
  - Audio transcription via Whisper
- **Conversation Features**
  - Real-time response streaming
  - Session persistence
  - Dynamic parameter controls (temperature, top-p)
  - Text-to-Speech (TTS) responses

### Supported Providers & Models
| Provider       | Models Supported                                                                 |
|----------------|----------------------------------------------------------------------------------|
| **OpenAI**     | GPT-4o, GPT-4 Turbo, DALL-E 3, Whisper, TTS                                     |
| **Anthropic**  | Claude 3.5 Sonnet, Claude 3.5 Haiku                                             |
| **Google**     | Gemini 1.5 Pro, Gemini 1.5 Flash                                               |
| **Ollama**     | Llama 3/3.2 Vision, Phi-3, Mistral, Mixtral, Deepseek R1 series                |
| **Groq**       | Llama 3 8B/70B, Mixtral 8x7B                                                   |
| **Cohere**     | Command, Command-R, Command-Light                                              |
| **OpenRouter** | Qwen2.5-coder, Mistral 7B                                                      |

### Advanced Features
- **Dynamic Conversation Routing**
  - Semantic understanding of queries
  - Automatic routing to appropriate services (chat, vision, image gen)
- **Assistant Framework**
  - Code interpreter for math/problem solving
  - File attachments support (PDF, CSV, images)
  - Custom tool integrations
- **Local Model Support**
  - Ollama integration for local inference
  - Vision capabilities with local images

## 📸 Screenshots

| Multi-Provider Interface | Assistant Conversation |
|--------------------------|------------------------|
| ![Multi LLM Providers](./src/resources/screenshot/1.jpg) | ![Assistant Chat](./src/resources/screenshot/2.jpg) |

## 🛠️ Getting Started

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download) (for local models)
- API keys for cloud providers

### Installation
```bash
# Clone repository
git clone https://github.com/vinhnx/VT.ai.git
cd VT.ai

# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your API keys to .env
```

### Configuration
```ini
# Example .env configuration
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
ASSISTANT_ID=your_assistant_id  # Optional
OLLAMA_HOST=http://localhost:11434  # For local models
```

## ⚙️ Advanced Configuration

### Model Settings
Configure in `llm_settings_config.py`:
```python
MODEL_ALIAS_MAP = {
    # Customize model aliases here
    "OpenAI - GPT-4o": "gpt-4o",
    "Ollama - Llama 3": "ollama/llama3",
    # ... other model definitions
}
```

### Chat Profiles
Two main interaction modes:
1. **Standard Chat**: Standard multi-LLM conversations
2. **Assistant Mode**: 
   - Code interpreter for complex problem solving
   - File attachments support
   - Persistent conversation threads

### Vision Processing
```python
# Example vision processing configuration
VISION_MODEL_MAP = {
    "Google - Gemini 1.5 Pro": "gemini/gemini-1.5-pro-latest",
    "OpenAI - GPT-4o": "gpt-4o"
}
```

## 🤝 Contributing

### Development Setup
1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Code standards:
- Follow PEP8 guidelines
- Include type hints for new features
- Add unit tests for critical components

### Testing
```bash
# Run basic tests
pytest tests/

# Verify model integrations
python -m pytest tests/integration/
```

## 📜 License

Distributed under MIT License. See [LICENSE](./LICENSE) for details.

## 📬 Connect
- Project Updates: [@vtdotai](https://twitter.com/vtdotai)
- Creator: [@vinhnx](https://twitter.com/vinhnx)
- GitHub: [vinhnx/VT.ai](https://github.com/vinhnx/VT.ai)
