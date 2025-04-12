# Router API Reference

This page documents the router module of VT.ai (`vtai/router/`), which provides semantic routing capabilities.

## Overview

The router module is responsible for analyzing user queries, determining their intent, and routing them to the appropriate handlers. It uses vector embeddings to understand the semantic meaning of queries, making it more flexible and robust than simple keyword matching.

## Key Components

### SemanticRouter Class

```python
class SemanticRouter:
    """
    A semantic router that uses vector embeddings to route queries to appropriate handlers.
    """

    def __init__(self, routes, embedding_model="BAAI/bge-small-en-v1.5", threshold=0.7):
        """
        Initialize the semantic router.

        Args:
            routes: List of Route objects defining routing patterns
            embedding_model: Model to use for embeddings
            threshold: Minimum similarity threshold for routing
        """
        # ...

    async def route(self, query, context=None):
        """
        Route a query to the appropriate handler.

        Args:
            query: User query to route
            context: Optional context dictionary

        Returns:
            Routing result with handler and metadata
        """
        # ...
```

### Route Class

```python
class Route:
    """
    Defines a routing destination with intent and handler.
    """

    def __init__(self, name, description, handler, examples=None):
        """
        Initialize a route.

        Args:
            name: Name of the intent
            description: Description of the intent
            handler: Function to handle the intent
            examples: Example queries for this intent
        """
        # ...
```

## Router Configuration

The router configuration is defined in `vtai/router/layers.json`, which contains intents and examples:

```json
{
  "intents": [
    {
      "name": "general_conversation",
      "description": "General conversation queries",
      "examples": [
        "Hello, how are you?",
        "What is the capital of France?",
        "Tell me about quantum physics",
        "..."
      ]
    },
    {
      "name": "image_generation",
      "description": "Requests to generate images",
      "examples": [
        "Generate an image of a mountain landscape",
        "Create a picture of a futuristic city",
        "Draw a cat playing with a ball of yarn",
        "..."
      ]
    },
    // Additional intents...
  ]
}
```

## Usage Examples

### Basic Routing

```python
# Import the router module
from vtai.router import SemanticRouter, Route

# Define route handlers
async def handle_general(message, context):
    # Handle general conversation
    pass

async def handle_image_gen(message, context):
    # Handle image generation
    pass

# Create routes
routes = [
    Route("general_conversation", "General conversation queries", handle_general),
    Route("image_generation", "Requests to generate images", handle_image_gen),
]

# Initialize the router
router = SemanticRouter(routes)

# Route a query
result = await router.route("Generate an image of a sunset")
# result will contain a reference to handle_image_gen and metadata
```

### Custom Embedding Model

```python
# Initialize with a custom embedding model
router = SemanticRouter(
    routes=routes,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    threshold=0.65
)
```

## Router Trainer

The router module includes a trainer for updating embeddings:

```python
# Import the trainer
from vtai.router.trainer import train_router

# Train the router with new examples
train_router(
    layers_file="path/to/layers.json",
    output_file="path/to/output.json"
)
```

## Best Practices

When working with the router:

1. **Adding New Intents**:
   - Include diverse examples for each intent
   - Ensure examples are distinct from other intents
   - Use at least 5-10 examples per intent for good coverage

2. **Performance Optimization**:
   - Cache embeddings for frequently used queries
   - Use smaller embedding models for faster inference
   - Adjust the similarity threshold based on your needs

3. **Error Handling**:
   - Implement a fallback route for unclassified queries
   - Log routing decisions for analysis
   - Periodically review and refine intent examples

## Source Code

For the complete source code of the router module, see the [GitHub repository](https://github.com/vinhnx/VT.ai/tree/main/vtai/router).
