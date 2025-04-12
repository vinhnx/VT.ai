# VT.ai Architecture

This document provides an overview of VT.ai's architecture, explaining how its components work together to create a powerful multimodal AI chat application.

## High-Level Architecture

VT.ai follows a modular architecture designed for flexibility and extensibility:

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

## Core Components

### 1. Entry Point (`vtai/app.py`)

The main application entry point handles initialization, manages the Chainlit web interface, and coordinates the conversation flow. It:

- Initializes the application configuration
- Sets up chat profiles and user sessions
- Manages message handling and routing
- Processes assistant runs and tool calls
- Handles user settings and TTS features

### 2. Routing Layer (`vtai/router/`)

The semantic routing system classifies user queries and directs them to appropriate handlers based on intent:

- **Encoder**: Uses FastEmbed with the BAAI/bge-small-en-v1.5 embedding model
- **Classification**: Performs vector similarity matching against predefined intents
- **Dynamic Dispatch**: Routes queries to specialized conversation handlers

### 3. Model Management

VT.ai uses LiteLLM as a unified interface to multiple AI providers:

- **Provider Abstraction**: Standardizes API calls across different model providers
- **Model Switching**: Allows seamless switching between models based on query needs
- **Error Handling**: Provides consistent error handling across providers

### 4. Utility Modules (`vtai/utils/`)

Various utility modules provide supporting functionality:

- **Configuration Management**: Handles API keys, settings, and environment variables
- **Conversation Handlers**: Processes different types of conversations (standard, thinking mode)
- **Media Processors**: Handles image analysis, generation, and TTS features
- **File Handlers**: Manages file uploads and processing
- **Error Handlers**: Provides consistent error handling and reporting

### 5. Assistant Tools (`vtai/assistants/`)

Implements specialized assistant capabilities:

- **Code Interpreter**: Executes Python code for data analysis
- **Thread Management**: Handles persistent conversation threads
- **Tool Processing**: Manages function calls and tool outputs

## Data Flow

When a user interacts with VT.ai, the following sequence occurs:

1. **User Input**: The user submits a message through the Chainlit web interface
2. **Message Processing**: The message is processed to extract text and handle any attached media
3. **Intent Classification**: The semantic router classifies the query intent using vector embeddings
4. **Model Selection**: Based on classification and settings, a specific model is selected
5. **Query Execution**: The query is sent to the appropriate model provider
6. **Response Processing**: The response is processed and may involve media generation
7. **UI Rendering**: The formatted response is displayed to the user in the web interface

## Configuration System

VT.ai's configuration system is layered and prioritized:

1. **Command-line arguments**: Highest priority, overrides other settings
2. **Environment variables**: Secondary priority, used if arguments not provided
3. **Configuration files**: `~/.config/vtai/.env` for persistent settings
4. **Default values**: Used when no specific settings are provided

## Extension Points

VT.ai is designed to be extensible in several ways:

1. **New Model Providers**: Add support for additional AI providers
2. **Custom Intents**: Extend the semantic router with new intent classifications
3. **Specialized Handlers**: Create custom handlers for specific conversation types
4. **New Assistant Tools**: Add new tools for specialized tasks

For details on extending VT.ai, see the [Extending VT.ai](extending.md) guide.
