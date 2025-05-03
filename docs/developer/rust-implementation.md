# Rust Implementation

This guide provides detailed information about VT.ai's Rust implementation, which offers a high-performance alternative to the Python version.

## Overview

The Rust implementation of VT.ai is designed with performance, reliability, and efficiency as primary goals. It provides the same core functionality as the Python version but with architectural choices that optimize for speed and resource usage.

Key benefits of the Rust implementation:

- **Performance**: Significantly faster response times and lower latency
- **Resource Efficiency**: Lower memory usage and CPU consumption
- **Concurrency**: Better handling of multiple simultaneous requests
- **Reliability**: Reduced likelihood of runtime errors through Rust's static typing
- **Security**: Memory safety guarantees from Rust's ownership model

## Architecture

The Rust implementation follows a similar high-level architecture to the Python version but with performance-focused components:

```
┌─────────────────┐     ┌───────────────┐     ┌────────────────┐
│ Web Server      │     │ Semantic      │     │ Model          │
│ (Axum)          │────▶│ Router        │────▶│ Providers      │
└─────────────────┘     └───────────────┘     └────────────────┘
        │                       │                     │
        ▼                       ▼                     ▼
┌─────────────────┐     ┌───────────────┐     ┌────────────────┐
│ WebSocket       │     │ Tool          │     │ Configuration  │
│ Handlers        │     │ Registry      │     │ Management     │
└─────────────────┘     └───────────────┘     └────────────────┘
```

### Core Components

#### Web Server (`src/app/`)

Built on the Axum web framework, the web server provides:

- **HTTP API**: REST endpoints for configuration and metrics
- **WebSocket Server**: Real-time bidirectional communication for chat
- **Static File Serving**: Delivers the web interface assets
- **Authentication**: API key validation and session management

#### Routing System (`src/router/`)

The semantic routing system efficiently classifies and directs user queries:

- **Intent Classification**: Uses efficient embeddings for query classification
- **Router Registry**: Trait-based system for pluggable routers
- **Dynamic Dispatch**: Fast routing to appropriate handlers
- **Caching**: Optimized caching of classification results

#### Model Client (`src/models/`)

Provides a unified client interface to multiple AI providers:

- **Provider Traits**: Abstract provider-specific implementations
- **Streaming Optimization**: Efficient handling of streaming responses
- **Batching**: Request batching for improved throughput
- **Rate Limiting**: Built-in rate limit handling and backoff

#### Tool Registry (`src/tools/`)

Manages the available tools and their execution:

- **Tool Trait**: Common interface for all tools
- **Safe Execution**: Sandboxed tool execution
- **Result Streaming**: Progressive result streaming
- **Extension API**: Easily add custom tools

#### Configuration (`src/config/`)

Handles application configuration with Rust-native approaches:

- **Structured Config**: Type-safe configuration with validation
- **Environment Variables**: Configuration through environment
- **Command Line Args**: Argument parsing with Clap
- **Config Files**: Persistent configuration storage

## Installation

### Prerequisites

- Rust toolchain (1.77.0 or newer recommended)
- Cargo package manager
- API keys for at least one LLM provider

### From Source

```bash
# Clone the repository
git clone https://github.com/vinhnx/VT.ai.git
cd VT.ai/rust-vtai

# Build the application
cargo build --release

# Run the application
./target/release/vtai
```

### Quick Start with run.sh

The included `run.sh` script simplifies building and running:

```bash
# Run with default settings
./run.sh

# Run with specific API key and model
./run.sh --api-key openai=sk-your-key-here --model o3-mini
```

### Docker Installation

```bash
# Build the Docker image
docker build -t vtai-rust .

# Run the container
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-your-key-here vtai-rust
```

## Configuration

The Rust implementation shares the same configuration directory (`~/.config/vtai/`) with the Python version for consistency. However, it uses Rust-native methods for configuration.

### Command Line Options

```bash
# View available options
./target/release/vtai --help

# Set API key
./target/release/vtai --api-key openai=sk-your-key-here

# Select model
./target/release/vtai --model o3-mini

# Set host and port
./target/release/vtai --host 127.0.0.1 --port 8080
```

### Environment Variables

```bash
# Set API keys
export OPENAI_API_KEY=sk-your-key-here
export ANTHROPIC_API_KEY=sk-ant-your-key-here

# Set default model
export VT_DEFAULT_MODEL=sonnet

# Configure server
export VT_HOST=0.0.0.0
export VT_PORT=9000
```

## Performance Comparison

The Rust implementation offers significant performance advantages over the Python version:

| Metric                    | Rust Implementation | Python Implementation |
|---------------------------|---------------------|----------------------|
| Initial response latency  | ~300ms              | ~800ms               |
| Memory usage (baseline)   | ~30MB               | ~150MB               |
| Max concurrent users      | ~500                | ~100                 |
| Startup time              | <1s                 | ~3s                  |

*Note: Exact performance depends on hardware, model selection, and query complexity.*

## Feature Compatibility

While the Rust implementation aims for feature parity with the Python version, some features may have implementation differences or limitations:

| Feature                   | Status             | Notes                                   |
|---------------------------|--------------------|-----------------------------------------|
| Text chat                 | ✅ Full support   | Complete implementation                  |
| Image analysis            | ✅ Full support   | Vision models fully supported            |
| Image generation          | ✅ Full support   | All image generation options available   |
| Voice interaction         | ⚠️ Partial support | Basic TTS, advanced features in progress |
| Web search                | ✅ Full support   | Tavily integration complete              |
| Thinking mode             | ✅ Full support   | Full reasoning capabilities              |
| Assistant mode            | ⚠️ Partial support | Code interpreter in development          |
| Dynamic routing           | ✅ Full support   | Efficient semantic routing               |

## Extending the Rust Implementation

### Adding a New Model Provider

To add a new model provider:

1. Implement the `ModelProvider` trait in `src/models/providers/`
2. Register the provider in the provider factory
3. Add the provider's configuration to the config module

Example provider implementation:

```rust
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyProviderConfig {
    pub api_key: String,
    pub base_url: Option<String>,
}

pub struct MyProvider {
    config: MyProviderConfig,
    client: reqwest::Client,
}

#[async_trait]
impl ModelProvider for MyProvider {
    async fn generate_text(&self, params: &CompletionParams) -> Result<CompletionResponse> {
        // Implementation details...
    }

    // Implement other required methods...
}
```

### Creating a Custom Tool

To create a custom tool:

1. Implement the `Tool` trait in a new module under `src/tools/`
2. Register the tool in the tool registry
3. Add any configuration needed for your tool

Example tool implementation:

```rust
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyToolConfig {
    pub api_key: Option<String>,
}

pub struct MyTool {
    config: MyToolConfig,
}

#[async_trait]
impl Tool for MyTool {
    fn name(&self) -> &'static str {
        "my_custom_tool"
    }

    fn description(&self) -> &'static str {
        "A custom tool that does something useful"
    }

    async fn execute(&self, params: Value) -> Result<Value> {
        // Tool implementation...
    }

    // Implement other required methods...
}
```

### Adding a New Router

To add a new semantic router:

1. Implement the `Router` trait in `src/router/`
2. Register your router in the router registry
3. Add configuration options if needed

Example router implementation:

```rust
use async_trait::async_trait;

pub struct MyCustomRouter {
    // Router fields...
}

#[async_trait]
impl Router for MyCustomRouter {
    async fn classify(&self, query: &str) -> Result<RouteIntent> {
        // Classification logic...
    }

    // Implement other required methods...
}
```

## Troubleshooting

### Common Issues

#### Build Errors

If you encounter build errors:

```bash
# Update Rust toolchain
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

#### API Connection Issues

For API connection problems:

```bash
# Check API key configuration
echo $OPENAI_API_KEY

# Test API connection
curl -s -X POST https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"gpt-3.5-turbo", "messages":[{"role":"user","content":"Hello"}]}'
```

#### Performance Optimization

To optimize performance:

```bash
# Enable release optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run with thread pool configuration
VT_THREAD_POOL_SIZE=8 ./target/release/vtai
```

## Future Development

The Rust implementation roadmap includes:

- **Full Assistant API Integration**: Complete implementation of OpenAI's Assistant API
- **Enhanced Tool Framework**: More powerful and flexible tool ecosystem
- **UI Improvements**: Custom web interface optimized for the Rust backend
- **Embedded Database**: Efficient local storage for conversation history
- **Distributed Deployment**: Multi-node deployment for high availability

## Contributing

Contributions to the Rust implementation are welcome:

1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Submit a pull request

For development setup:

```bash
# Setup development environment
cd rust-vtai

# Install development tools
rustup component add clippy rustfmt

# Run tests
cargo test

# Check code style
cargo fmt --check
cargo clippy
```

---

For more information on the architectural differences between the Python and Rust implementations, see the [Architecture Overview](architecture.md).
