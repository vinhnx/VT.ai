# VT.ai (Rust Version)

A Rust implementation of the VT.ai multimodal AI chat application with dynamic conversation routing.

## Overview

This project is a Rust port of the original Python-based VT.ai application. It provides a minimal, yet powerful chat interface with semantic routing capabilities, tool integration, and LLM provider abstraction.

## Features

- **Semantic Routing**: Intelligently routes user queries to appropriate handlers
- **Multiple LLM Support**: Works with OpenAI, Anthropic, DeepSeek and more
- **Tool Integration**: Includes code interpreter, file processing, and search capabilities
- **Web Interface**: Axum-based web server with WebSocket support for real-time chat
- **OpenAI Assistant API**: Full integration with OpenAI's Assistant API
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
   git clone https://github.com/yourusername/rust-vtai.git
   cd rust-vtai
   ```

2. Build the application:

   ```
   cargo build --release
   ```

### Usage

Run the application with:

```
cargo run --release
```

Or use the compiled binary:

```
./target/release/vtai
```

### Configuration

You can provide API keys and select a model when starting the application:

```
./target/release/vtai --api-key openai=sk-your-api-key-here --model o3-mini
```

Supported models include:

- `gpt-4.1-mini`: OpenAI - GPT-4.1 Mini
- `gpt-4o-mini`: OpenAI - GPT-4o Mini
- `gpt-4o`: OpenAI - GPT-4o
- `gpt-4.1`: OpenAI - GPT-4.1
- `gpt-4.1-nano`: OpenAI - GPT-4.1 Nano
- `o1`: OpenAI - GPT-o1
- `o3-mini`: OpenAI - GPT-o3 Mini
- `o1-mini`: OpenAI - GPT-o1 Mini
- `o1-pro`: OpenAI - GPT-o1 Pro
- `claude-3-7-sonnet-20250219`: Anthropic - Claude 3.7 Sonnet
- `claude-3-5-sonnet-20241022`: Anthropic - Claude 3.5 Sonnet
- `claude-3-5-haiku-20241022`: Anthropic - Claude 3.5 Haiku
- `deepseek/deepseek-reasoner`: DeepSeek R1
- `deepseek/deepseek-chat`: DeepSeek V3
- `deepseek/deepseek-coder`: DeepSeek Coder
- `gemini/gemini-2.0-pro`: Google - Gemini 2.0 Pro
- `gemini/gemini-2.0-flash`: Google - Gemini 2.0 Flash
- `gemini/gemini-2.0-flash-exp`: Google - Gemini 2.0 Flash Exp
- ...and more, see src/utils/models_map.rs for the full list.

## Dynamic Model Dropdown

The model selection dropdown in the web UI is now dynamically populated from the backend. The backend serves the available models from `src/utils/models_map.rs` via the `/api/models` endpoint.

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

**Note:** If you want to use custom models or providers, add them to `MODEL_ALIAS_MAP` and ensure your backend is configured to route requests appropriately.

## Development

This is a work in progress. The current implementation includes:

- Basic web server setup with WebSocket support
- Core semantic routing functionality (simplified)
- Tool registry and implementations
- OpenAI Assistant API integration

Future work includes:

- Improved embedding-based routing
- Advanced tool integrations
- UI enhancements
- More comprehensive testing

## License

MIT
