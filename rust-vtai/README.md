# VT.ai (Rust Version)

A Rust implementation of the VT.ai multimodal AI chat application with dynamic conversation routing.

## Overview

This project is a Rust port of the original Python-based VT.ai application. It provides a minimal, yet powerful chat interface with semantic routing capabilities, tool integration, and LLM provider abstraction.

## Features

- **Semantic Routing**: Intelligently routes user queries to appropriate handlers
- **Multiple LLM Support**: Works with OpenAI, Anthropic, DeepSeek, Google, Meta, Mistral, Groq, Cohere and more
- **Tool Integration**: Includes code interpreter, file processing, and search capabilities
- **Web Interface**: Axum-based web server with WebSocket support for real-time chat
- **OpenAI Assistant API**: Full integration with OpenAI's Assistant API
- **Thinking Mode**: Support for step-by-step reasoning with the <think> tag on supported models
- **Vision Capabilities**: Support for image analysis with compatible models
- **Error Handling**: Robust error handling and logging

## Project Structure

- `src/app`: Web server and chat functionality
- `src/router`: Semantic routing for conversation classification
- `src/tools`: Tool implementations (code execution, file operations, search)
- `src/assistants`: OpenAI Assistant API integration
- `src/utils`: Shared utilities, error handling, and configuration

## Getting Started

### Prerequisites

- Rust toolchain (1.77.0 or newer recommended)
- API keys for at least one LLM provider (OpenAI, Anthropic, etc.)

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/vinhnx/VT.ai.git
   cd VT.ai/rust-vtai
   ```

2. Build the application:

   ```
   cargo build --release
   ```

### Usage

You can run the application using the provided convenience script:

```bash
# Using the convenience script (recommended)
./run.sh

# With command-line arguments
./run.sh --api-key openai=sk-your-api-key-here --model o3-mini
```

Or run it manually:

```
cargo run --release
```

Or use the compiled binary directly:

```
./target/release/vtai
```

### Configuration

You can provide API keys and select a model when starting the application:

```
./run.sh --api-key openai=sk-your-api-key-here --model o3-mini
```

Supported models include a wide range of options from providers like:

- **OpenAI**: GPT-4o, GPT-4.1, GPT-o1, GPT-o3 Mini, etc.
- **Anthropic**: Claude 3.7 Sonnet, Claude 3.5 Sonnet/Haiku, etc.
- **DeepSeek**: DeepSeek Reasoner (R1), DeepSeek Chat (V3), DeepSeek Coder
- **Google**: Gemini 2.0 Pro, Gemini 2.0 Flash, etc.
- **Mistral**: Mistral Small, Mistral Large
- **Groq**: Llama 4 Scout, Llama 3, Mixtral 8x7b
- **Cohere**: Command, Command-R, Command-Light, Command-R-Plus
- **Meta**: Llama 4 Maverick, Llama 4 Scout (via OpenRouter)
- **Qwen**: QWQ 32B, Qwen 2.5 VL/Coder (via OpenRouter)
- **Ollama**: Local models including DeepSeek R1, Qwen2.5-coder, Llama 3/3.1/3.2, Phi-3, etc.

See `src/utils/models_map.rs` for the full list of supported models.

## Dynamic Model Dropdown

The model selection dropdown in the web UI is dynamically populated from the backend. The backend serves the available models from `src/utils/models_map.rs` via the `/api/models` endpoint.

**To add or remove models:**
- Edit the `MODEL_ALIAS_MAP` in `src/utils/models_map.rs`.
- Rebuild and restart the Rust backend.
- The UI will automatically reflect the changes.

**How it works:**
- The frontend fetches `/api/models` on page load and populates the dropdown.
- The backend endpoint returns a list of `{ label, value }` pairs based on `MODEL_ALIAS_MAP`.

**Example:**
```rust
// src/utils/models_map.rs
pub static MODEL_ALIAS_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("OpenAI - GPT-4o Mini", "gpt-4o-mini");
    // ...
    map
});
```

## Special Features

### Thinking Mode

Supported models can use the `<think>` tag to show their step-by-step reasoning. This feature is available on models listed in the `REASONING_MODELS` list in `src/utils/models_map.rs`.

### Vision Capabilities

The Rust implementation supports image analysis with compatible models like OpenAI's GPT-4o, Google's Gemini, and Llama 3.2 Vision (via Ollama).

## Development

This is a work in progress. The current implementation includes:

- Basic web server setup with WebSocket support
- Core semantic routing functionality
- Tool registry and implementations
- OpenAI Assistant API integration
- Support for a wide range of LLM providers

Future work includes:

- Improved embedding-based routing
- Advanced tool integrations
- UI enhancements
- More comprehensive testing

## License

MIT
