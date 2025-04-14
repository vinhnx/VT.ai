# Getting Started with VT.ai

This guide will help you get up and running with VT.ai quickly.

## Implementation Options

VT.ai is available in two implementations:

1. **Python Implementation**: The original implementation with full feature support
2. **Rust Implementation**: A high-performance port focused on efficiency and reliability

You can choose which implementation to use based on your needs. Both share similar configuration approaches and supported models.

## Python Installation Options

VT.ai can be installed and run in multiple ways depending on your needs:

### Quick Install from PyPI

```bash
# Install VT.ai from PyPI
pip install vtai
```

### Quick Start with uvx (No Installation)

If you have [uv](https://github.com/astral-sh/uv) installed, you can try VT.ai without installing it permanently:

```bash
# Set your API key in the environment
export OPENAI_API_KEY='sk-your-key-here'

# Run VT.ai directly using uvx
uvx vtai
```

This creates a temporary virtual environment just for this session. When you're done, nothing is left installed on your system.

### Installation with uv

```bash
# If you need to install uv first
python -m pip install uv

# Install VT.ai with uv
uv tool install --force --python python3.11 vtai@latest
```

### Installation with pipx

```bash
# If you need to install pipx first
python -m pip install pipx

# Install VT.ai with pipx
pipx install vtai
```

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

## Rust Implementation Setup

The Rust implementation provides an alternative high-performance version:

### Prerequisites

- Rust toolchain (1.77.0 or newer recommended)
- API keys for at least one LLM provider

### Installation Steps

```bash
# Clone repository if you haven't already
git clone https://github.com/vinhnx/VT.ai.git
cd VT.ai/rust-vtai
```

### Quick Start with run.sh (Recommended)

The Rust implementation includes a convenient shell script that handles building and running the application:

```bash
# Run the application using the convenience script
./run.sh

# With API key and model selection
./run.sh --api-key openai=sk-your-key-here --model o3-mini
```

The `run.sh` script:
- Builds the application in release mode
- Checks for API keys in environment variables
- Runs the application with any provided command-line arguments
- Displays helpful messages and warnings

### Manual Build and Run

If you prefer to build and run manually:

```bash
# Build the application
cargo build --release

# Run the application
./target/release/vtai
```

## API Key Configuration

You'll need at least one API key to use VT.ai effectively. You can set your API keys in several ways:

### Python Implementation

#### Command Line Option

```bash
# Set OpenAI API key
vtai --api-key openai=<your-key>
```

#### Environment Variables

```bash
# For OpenAI (recommended for first-time users)
export OPENAI_API_KEY='sk-your-key-here'
vtai

# For Anthropic Claude models
export ANTHROPIC_API_KEY='sk-ant-your-key-here'
vtai --model sonnet

# For Google Gemini models
export GEMINI_API_KEY='your-key-here'
vtai --model gemini-2.5
```

### Rust Implementation

The Rust implementation uses a similar configuration approach:

```bash
# Set OpenAI API key
./target/release/vtai --api-key openai=<your-key>

# Select a specific model
./target/release/vtai --model o3-mini
```

API keys are saved to the same configuration directory as the Python implementation for consistency.

## First Run Experience

When you run VT.ai for the first time:

1. The application will create a configuration directory at `~/.config/vtai/`
2. It will download necessary model files (tokenizers, embeddings, etc.)
3. The web interface will open at [http://localhost:8000](http://localhost:8000)
4. If no API keys are configured, you'll be prompted to add them

To ensure the best first-run experience:

```bash
# Set at least one API key before running (OpenAI recommended for beginners)
export OPENAI_API_KEY='sk-your-key-here'

# Run the application (Python implementation)
vtai

# Or for the Rust implementation
cd rust-vtai
cargo run --release
```

## Basic Usage

After starting VT.ai, you'll be presented with a chat interface. Here are some basic operations:

1. **Standard Chat**: Type a message and press Enter to send it.
2. **Image Analysis**: Upload an image or provide a URL to analyze it.
3. **Image Generation**: Type a prompt like "Generate an image of a mountain landscape" to create an image.
4. **Thinking Mode**: Use the `<think>` tag to see the model's step-by-step reasoning.
5. **Voice Interaction**: Enable voice features to interact with speech.

For more detailed usage instructions, see the [Features](features.md) page.

## Upgrading VT.ai

### Upgrading the Python Implementation

To upgrade VT.ai to the latest version:

```bash
# If installed with pip
pip install --upgrade vtai

# If installed with pipx
pipx upgrade vtai

# If installed with uv
uv tool upgrade vtai
```

### Upgrading the Rust Implementation

To upgrade the Rust implementation:

```bash
# Navigate to the repository
cd VT.ai

# Pull the latest changes
git pull

# Build the latest version
cd rust-vtai
cargo build --release
```

## Next Steps

- Explore the [Features](features.md) documentation to learn about all capabilities
- Learn about [Configuration](configuration.md) options
- Check out the [Models](models.md) documentation to understand different model options
- Visit [Troubleshooting](troubleshooting.md) if you encounter any issues
- Review the [Architecture](../developer/architecture.md) documentation to understand how VT.ai works internally
