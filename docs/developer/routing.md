# Semantic Routing in VT.ai

This document explains VT.ai's semantic routing system, which intelligently directs user queries to specialized handlers based on their intent.

## Overview

The semantic routing system is a key component of VT.ai that analyzes user queries and automatically determines the most appropriate handler to process them. Unlike simple keyword matching, this system uses vector embeddings to understand the semantic meaning of queries, making it more robust and flexible.

## How It Works

The routing process follows these steps:

1. **Query Embedding**: The user's query is converted into a vector representation (embedding) using the BAAI/bge-small-en-v1.5 model via FastEmbed.

2. **Intent Matching**: The query embedding is compared against predefined intent embeddings using cosine similarity to find the closest match.

3. **Handler Selection**: Based on the matched intent, the system selects the appropriate handler function to process the query.

4. **Response Generation**: The selected handler processes the query and generates a response, which may involve calling specific models or executing specialized logic.

## Key Components

### Router Module

The router module is located in `vtai/router/` and consists of:

- **`__init__.py`**: Core routing functionality
- **`constants.py`**: Routing-related constants
- **`layers.json`**: Intent definitions and examples
- **`trainer.py`**: Utility for training the router with new intents

### Intent Definitions

Intents are defined in the `layers.json` file with the following structure:

```json
{
  "intents": [
    {
      "name": "intent_name",
      "description": "Description of what this intent handles",
      "examples": [
        "Example query 1",
        "Example query 2",
        "Example query 3"
      ]
    },
    // More intents...
  ]
}
```

Each intent includes:

- A unique name
- A description of what it handles
- Example queries that match this intent

### Embedding Model

VT.ai uses the BAAI/bge-small-en-v1.5 embedding model through FastEmbed, which provides:

- High-quality semantic vector representations
- Efficient computation for low-latency routing
- Good performance across multiple languages

### Handler Functions

Handler functions are defined in `vtai/utils/conversation_handlers.py` and are connected to intents in the router configuration. Each handler:

- Takes the user message and conversation history as input
- Processes the query according to its specialized logic
- Generates an appropriate response
- Sends the response back to the user

## Default Intents

VT.ai includes several predefined intents:

1. **General Conversation**: For standard chat interactions
2. **Image Generation**: For creating images from text descriptions
3. **Vision Analysis**: For analyzing and interpreting images
4. **Thinking Mode**: For accessing step-by-step reasoning from models
5. **Code Assistance**: For programming help and code execution
6. **Data Analysis**: For working with data and performing calculations

## Customizing the Router

The semantic router can be extended with new intents. The process involves:

1. **Adding Intent Definitions**: Update `vtai/router/layers.json` with new intents and examples
2. **Training the Router**: Run `python -m vtai.router.trainer` to update embeddings
3. **Creating Handler Functions**: Implement specialized handlers in `vtai/utils/conversation_handlers.py`
4. **Updating Router Configuration**: Connect intents to handlers in the router initialization

For detailed instructions, see the [Extending VT.ai](extending.md) guide.

## Dynamic Routing Control

VT.ai allows users to control the routing behavior:

- **Enable/Disable**: Users can toggle dynamic routing in the settings menu
- **Override**: Users can select specific models to bypass routing
- **Force Routing**: Adding specific markers to messages can force certain handlers

## Performance Considerations

The semantic router is designed to be efficient, but there are some considerations:

- **Embedding Computation**: The initial embedding of intents happens at startup
- **Query Embedding**: Each user query needs to be embedded before routing
- **Model Loading**: The embedding model is loaded into memory at startup
- **Cache Usage**: Frequent queries may benefit from embedding caching

## Technical Details

### Embedding Process

The technical process of embedding a query involves:

```python
# Pseudocode for query embedding
from fastembed import TextEmbedding

# Load the model (done at initialization)
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

# Embed the query
query_embedding = embedding_model.embed(query)

# Compare to intent embeddings
similarities = [cosine_similarity(query_embedding, intent_embedding)
                for intent_embedding in intent_embeddings]

# Get the best match
best_match_index = np.argmax(similarities)
best_intent = intents[best_match_index]
```

### Routing Decision Logic

The routing decision is made based on:

1. The closest matching intent
2. A confidence threshold to avoid mis-routing
3. User preferences and settings
4. Fallback logic for when no clear match is found

## Troubleshooting

If you encounter issues with the routing system:

- **Misrouted Queries**: Add more examples to the relevant intent
- **Unhandled Intents**: Check if you need to create a new intent
- **Slow Routing**: Ensure embeddings are properly cached
- **Failed Routing**: Verify the embedding model is correctly loaded
