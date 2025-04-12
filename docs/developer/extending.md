# Extending VT.ai

This guide explains how to extend VT.ai with new capabilities. VT.ai is designed to be modular and extensible, allowing developers to add new features, models, and integrations.

## Adding New Model Providers

VT.ai uses LiteLLM to abstract model providers, making it relatively straightforward to add support for new AI models.

### Steps to Add a New Provider

1. **Update Provider Configuration**

   Edit `vtai/utils/llm_providers_config.py` to add your new provider:

   ```python
   # Add to the PROVIDERS dictionary
   PROVIDERS = {
       # ...existing providers...
       "new_provider": {
           "name": "New Provider",
           "models": ["new-model-1", "new-model-2"],
           "env_var": "NEW_PROVIDER_API_KEY",
           "icon": "new_provider.png",  # Add icon to vtai/resources/
       }
   }
   ```

2. **Add Provider Icon**

   Place the provider's icon in `vtai/resources/` directory.

3. **Update Model Mappings**

   Update the model mappings in the same file to include your new models:

   ```python
   # Add to MODEL_PROVIDER_MAP
   MODEL_PROVIDER_MAP.update({
       "new-model-1": "new_provider",
       "new-model-2": "new_provider",
   })
   ```

4. **Implement Special Handling (if needed)**

   If your provider requires special handling, add custom logic to the appropriate utility files:

   - For conversation handling: `vtai/utils/conversation_handlers.py`
   - For media processing: `vtai/utils/media_processors.py`

## Extending the Semantic Router

The semantic router classifies user queries to determine the best handler. You can add new intents to the router for specialized handling.

### Steps to Add a New Intent

1. **Update Routing Layers**

   Edit `vtai/router/layers.json` to add your new intent:

   ```json
   {
     "intents": [
       // ...existing intents...
       {
         "name": "my_new_intent",
         "description": "Description of what this intent handles",
         "examples": [
           "Example query 1",
           "Example query 2",
           "Example query 3"
         ]
       }
     ]
   }
   ```

2. **Train the Router**

   Run the router trainer to update the embeddings:

   ```bash
   python -m vtai.router.trainer
   ```

3. **Add Handler Function**

   Create a handler function in `vtai/utils/conversation_handlers.py`:

   ```python
   async def handle_my_new_intent(message, messages, client, **kwargs):
       """
       Handle queries matching my_new_intent.

       Args:
           message: The user message
           messages: Message history
           client: LLM client
           **kwargs: Additional arguments
       """
       # Your handler implementation here
       # ...

       # Send response
       await cl.Message(content=response).send()
   ```

4. **Update Router Configuration**

   Add your handler to the routing configuration in the app initialization:

   ```python
   # In vtai/utils/config.py
   route_layer = SemanticRouter(
       # ...existing configuration...
       routes=[
           # ...existing routes...
           Route(
               name="my_new_intent",
               description="Handles my new intent",
               handler=handle_my_new_intent,
           ),
       ]
   )
   ```

## Adding New Assistant Tools

You can extend VT.ai with new assistant tools for specialized capabilities.

### Steps to Add a New Tool

1. **Define Tool Interface**

   Create a new tool definition in `vtai/tools/` or extend an existing one:

   ```python
   # In vtai/tools/my_new_tool.py
   from typing import Dict, Any

   async def my_new_tool_function(args: Dict[str, Any]) -> Dict[str, Any]:
       """
       Implement a new tool capability.

       Args:
           args: Tool arguments

       Returns:
           Tool results
       """
       # Tool implementation
       # ...

       return {"result": "Output from the tool"}
   ```

2. **Register the Tool**

   Add your tool to the assistant configuration in `vtai/utils/assistant_tools.py`:

   ```python
   # Add to the tools list
   ASSISTANT_TOOLS = [
       # ...existing tools...
       {
           "type": "function",
           "function": {
               "name": "my_new_tool",
               "description": "Description of what this tool does",
               "parameters": {
                   "type": "object",
                   "properties": {
                       "param1": {
                           "type": "string",
                           "description": "Description of param1"
                       },
                       # Add more parameters as needed
                   },
                   "required": ["param1"]
               }
           }
       }
   ]
   ```

3. **Implement Tool Processing**

   Add a tool processor in `vtai/app.py` to handle the tool execution:

   ```python
   # Add to the process_function_tool logic to handle your tool
   if function_name == "my_new_tool":
       from vtai.tools.my_new_tool import my_new_tool_function
       result = await my_new_tool_function(function_args)
       return {
           "output": result,
           "tool_call_id": tool_call.id
       }
   ```

## Creating Custom UI Components

VT.ai uses Chainlit for the web interface. You can extend the UI with custom components.

### Steps to Add Custom UI Elements

1. **Define Custom Component**

   Create a custom component in a new file in the `vtai` directory:

   ```python
   # In vtai/ui/custom_component.py
   import chainlit as cl

   async def create_custom_component(data):
       """Create a custom UI component."""
       component = cl.Component(
           name="custom_component",
           props={"data": data},
           url="custom_component.js"  # Will need to create this
       )
       await component.send()
   ```

2. **Add Client-Side Implementation**

   If needed, create JavaScript for client-side functionality in `.chainlit/public/custom_component.js`.

3. **Integrate Component**

   Use your component in the appropriate part of the application, such as in conversation handlers.

## Testing Extensions

To test your extensions:

1. **Run Unit Tests**

   Create tests for your new functionality in the `tests` directory:

   ```bash
   # Run tests for your extension
   pytest tests/unit/test_my_extension.py
   ```

2. **Manual Testing**

   Run VT.ai in development mode to test your changes interactively:

   ```bash
   chainlit run vtai/app.py -w
   ```

## Best Practices

When extending VT.ai, follow these best practices:

1. **Maintain Backward Compatibility**: Ensure your changes don't break existing functionality
2. **Follow the Existing Pattern**: Maintain the project's coding style and architecture
3. **Add Tests**: Include tests for your new functionality
4. **Update Documentation**: Document your extensions in the codebase and in the docs directory
5. **Handle Errors Gracefully**: Implement proper error handling for your new features
6. **Optimize Performance**: Consider the performance impact of your changes
