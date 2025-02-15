---
<p align="center">
  <img src="./public/logo_dark.png" height="200" alt="VT.ai Logo" />
  <h1 align="center">VT.ai</h1>
  <p align="center">Multimodal AI Platform with Dynamic Routing & Assistant</p>
  
  [![Open in GitHub Codespaces](https://img.shields.io/badge/Open%20in-Codespaces-blue?logo=github)](https://codespaces.new/vinhnx/VT.ai)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Twitter Follow](https://img.shields.io/twitter/follow/vtdotai?style=social)](https://twitter.com/vtdotai)
</p>

## 🚀 Features

### Multi-Provider AI Orchestration
✅ **Supported Models Provider**:
- DeepSeek
- OpenAI
- Anthropic
- Google Gemini
- Local Models via Ollama (Llama3, Phi-3, Mistral, etc.)
- Cohere
- OpenRouter

✨ **Core Capabilities**:
- Dynamic conversation routing with SemanticRouter
- Multi-modal interactions (Text/Image/Audio)
- Assistant framework with code interpretation
- Real-time response streaming
- Cross-provider model switching
- Local model support with Ollama integration

## 📦 Installation

### Prerequisites
- Python 3.11+
- Ollama (for local models) - [Install Guide](https://ollama.com/download)

```bash
git clone https://github.com/vinhnx/VT.ai.git
cd VT.ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
cp .env.example .env
```

## 🔧 Configuration

Populate `.env` with your API keys:
```ini
OPENAI_API_KEY=sk-your-key
GEMINI_API_KEY=your-gemini-key
COHERE_API_KEY=your-cohere-key
ANTHROPIC_API_KEY=your-claude-key

# Local Models
OLLAMA_HOST=http://localhost:11434
```

## 🖥️ Usage

### Start Application
```bash
# Train semantic router (recommended)
python src/router/trainer.py

# Launch interface
chainlit run src/app.py -w
```

### Key Commands
| Shortcut | Action                          |
|----------|---------------------------------|
| Ctrl+/   | Switch model provider          |
| Ctrl+,   | Open settings                  |
| Ctrl+L   | Clear conversation history     |

## 🧩 Chat Profiles

### Standard Chat Mode
- Multi-LLM conversations
- Dynamic model switching
- Image generation & analysis
- Audio transcription

### Assistant Mode (Beta)
```python
# Example assistant capabilities
async def solve_math_problem(problem: str):
    assistant = MinoAssistant()
    return await assistant.solve(problem)
```
- Code interpreter for complex calculations
- File attachments (PDF/CSV/Images)
- Persistent conversation threads
- Custom tool integrations

## 🏗️ Project Structure

```
VT.ai/
├── src/
│   ├── assistants/       # Custom AI assistant implementations
│   ├── router/           # Semantic routing configuration
│   ├── utils/            # Helper functions & configs
│   └── app.py            # Main application entrypoint
├── public/               # Static assets
├── requirements.txt      # Python dependencies
└── .env.example          # Environment template
```

## 🌐 Supported Models

| Category       | Models                                                                 |
|----------------|-----------------------------------------------------------------------|
| **Chat**       | GPT-4o, Claude 3.5, Gemini 1.5, Llama3-70B, Mixtral 8x7B             |
| **Vision**     | GPT-4o, Gemini 1.5 Pro, Llama3.2 Vision                              |
| **Image Gen**  | DALL-E 3                                                             |
| **TTS**        | OpenAI TTS-1, TTS-1-HD                                               |
| **Local**      | Llama3, Phi-3, Mistral, Deepseek R1 series                           |

## 🤝 Contributing

### Development Setup
```bash
# Install development tools
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
```

### Contribution Guidelines
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Add Type hints for new functions
4. Update documentation
5. Open Pull Request

## 📄 License

MIT License - See [LICENSE](LICENSE) for full text.

## 🌟 Acknowledgements

- Inspired by [Chainlit](https://chainlit.io) for chat interface
- Powered by [LiteLLM](https://docs.litellm.ai) for model abstraction
- Semantic routing via [SemanticRouter](https://github.com/aurelio-labs/semantic-router)
