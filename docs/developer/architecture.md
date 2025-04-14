# VT.ai Architecture

This document provides an overview of VT.ai's architecture, explaining how its components work together to create a powerful multimodal AI chat application.

## Implementation Options

VT.ai is available in two implementations:

1. **Python Implementation**: The original implementation built with Python, Chainlit, and LiteLLM
2. **Rust Implementation**: A high-performance port focused on efficiency and reliability

This document covers both implementations, highlighting their architectural differences and similarities.

## Python Implementation

### High-Level Architecture

The Python implementation follows a modular architecture designed for flexibility and extensibility:

```
┌─────────────────┐     ┌───────────────┐     ┌────────────────┐
│ Web Interface   │     │ Semantic      │     │ Model          │
│ (Chainlit)      │────▶│ Router        │────▶│ Providers      │
└─────────────────┘     └───────────────┘     └────────────────┘
        │                       │                     │
        ▼                       ▼                     ▼
┌─────────────────┐     ┌───────────────┐     ┌────────────────┐
│ File/Media      │     │ Assistant     │     │ API Key        │
│ Processing      │     │ Tools         │     │ Management     │
└─────────────────┘     └───────────────┘     └────────────────┘
```

### Core Components

#### 1. Entry Point (`vtai/app.py`)

The main application entry point handles initialization, manages the Chainlit web interface, and coordinates the conversation flow. It:

- Initializes the application configuration
- Sets up chat profiles and user sessions
- Manages message handling and routing
- Processes assistant runs and tool calls
- Handles user settings and TTS features

#### 2. Routing Layer (`vtai/router/`)

The semantic routing system classifies user queries and directs them to appropriate handlers based on intent:

- **Encoder**: Uses FastEmbed with the BAAI/bge-small-en-v1.5 embedding model
- **Classification**: Performs vector similarity matching against predefined intents
- **Dynamic Dispatch**: Routes queries to specialized conversation handlers

#### 3. Model Management

The Python implementation uses LiteLLM as a unified interface to multiple AI providers:

- **Provider Abstraction**: Standardizes API calls across different model providers
- **Model Switching**: Allows seamless switching between models based on query needs
- **Error Handling**: Provides consistent error handling across providers

#### 4. Utility Modules (`vtai/utils/`)

Various utility modules provide supporting functionality:

- **Configuration Management**: Handles API keys, settings, and environment variables
- **Conversation Handlers**: Processes different types of conversations (standard, thinking mode)
- **Media Processors**: Handles image analysis, generation, and TTS features
- **File Handlers**: Manages file uploads and processing
- **Error Handlers**: Provides consistent error handling and reporting

#### 5. Assistant Tools (`vtai/assistants/`)

Implements specialized assistant capabilities:

- **Code Interpreter**: Executes Python code for data analysis
- **Thread Management**: Handles persistent conversation threads
- **Tool Processing**: Manages function calls and tool outputs

## Rust Implementation

### High-Level Architecture

The Rust implementation follows a similar architecture but with performance-focused components:

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

#### 1. Web Server (`src/app/`)

The entry point for the Rust implementation is based on the Axum web framework:

- **HTTP Server**: Handles REST API endpoints
- **WebSocket Server**: Manages real-time chat communication
- **Static Files**: Serves the web interface assets
- **API Endpoints**: Exposes model selection and configuration endpoints

#### 2. Routing System (`src/router/`)

The Rust implementation includes a semantic routing system similar to the Python version:

- **Intent Classification**: Uses embeddings for query classification
- **Router Registry**: Manages available routers and their capabilities
- **Dynamic Dispatch**: Routes requests to appropriate handlers

#### 3. Tool Integration (`src/tools/`)

The Rust implementation provides a robust tool integration system:

- **Tool Registry**: Central registry for available tools
- **Code Execution**: Safely runs code in isolated environments
- **File Operations**: Handles file uploads and processing
- **Search Tools**: Provides search capabilities within conversations

#### 4. Assistant Integration (`src/assistants/`)

Implements the OpenAI Assistant API integration:

- **Assistant Management**: Creates and manages assistants
- **Thread Handling**: Manages conversation threads
- **Tool Invocation**: Handles tool calls from the assistant

#### 5. Utilities (`src/utils/`)

Common utilities shared across the application:

- **Error Handling**: Comprehensive error types and handling
- **Configuration**: Environment-based configuration
- **Model Management**: Model alias resolution and provider abstraction
- **Authentication**: API key management and validation

## Data Flow Comparison

Both implementations follow a similar data flow pattern with some implementation-specific differences:

### Python Implementation

1. **User Input**: User submits a message through the Chainlit web interface
2. **Message Processing**: The message is processed to extract text and media
3. **Intent Classification**: The semantic router classifies the query intent
4. **Model Selection**: A specific model is selected based on classification
5. **Query Execution**: The query is sent to the appropriate model provider via LiteLLM
6. **Response Processing**: The response is processed (may include media generation)
7. **UI Rendering**: The formatted response is displayed in the web interface

### Rust Implementation

1. **User Input**: User submits a message through the web interface over WebSocket
2. **Message Processing**: The WebSocket handler processes the incoming message
3. **Intent Classification**: The router classifies the message intent
4. **Model Selection**: The appropriate model is selected based on the classification
5. **Query Execution**: The query is sent directly to the model provider API
6. **Response Streaming**: The response is streamed back to the client via WebSocket
7. **UI Rendering**: The client-side JavaScript renders the response in the interface

## Configuration System

Both implementations use a similar configuration approach with some implementation differences:

### Python Implementation

Uses a layered configuration system with priorities:

1. **Command-line arguments**: Highest priority
2. **Environment variables**: Secondary priority
3. **Configuration files**: `~/.config/vtai/.env` for persistence
4. **Default values**: Used when no specific settings are provided

### Rust Implementation

Uses a Rust-native configuration approach:

1. **Command-line arguments**: Processed using Clap
2. **Environment variables**: Read using the dotenv crate
3. **Configuration files**: Similar structure to Python implementation
4. **Default values**: Defined using Rust's Option and Default traits

## Extension Points

Both implementations provide extension points for customization:

### Python Implementation

1. **New Model Providers**: Add support for additional AI providers
2. **Custom Intents**: Extend the semantic router with new intent classifications
3. **Specialized Handlers**: Create custom handlers for specific conversation types
4. **New Assistant Tools**: Add new tools for specialized tasks

### Rust Implementation

1. **New Model Providers**: Implement new provider traits
2. **Custom Routers**: Add new router implementations to the registry
3. **Tool Extensions**: Implement the Tool trait for new functionality
4. **Middleware Components**: Add new middleware for request/response processing

For details on extending either implementation, see the [Extending VT.ai](extending.md) guide.
