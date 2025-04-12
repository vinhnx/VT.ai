# API Reference

This section provides detailed reference documentation for VT.ai's components and modules. The documentation is generated directly from the codebase and includes information about classes, methods, functions, and their parameters.

## Core Components

VT.ai is organized into several core components:

### App Module

The App module is the main entry point for the VT.ai application. It handles initialization, manages the Chainlit web interface, and coordinates the flow of conversations.

[View App Documentation](app.md)

### Router Module

The Router module contains the semantic routing logic that classifies user queries and directs them to appropriate handlers. It uses FastEmbed to encode queries and match them to predefined intents.

[View Router Documentation](router.md)

### Utils Module

The Utils module provides various utility functions and classes that support the core functionality of VT.ai, including configuration management, conversation handling, and media processing.

[View Utils Documentation](utils.md)

### Assistants Module

The Assistants module implements specialized AI assistants with capabilities like code interpretation, file processing, and function calling.

[View Assistants Documentation](assistants.md)

## Using the API Reference

Each page in the API reference includes:

- Class and function definitions
- Parameter descriptions
- Return value information
- Usage examples where available
- Source code links

This documentation is intended for developers who want to understand, extend, or modify VT.ai's functionality.
