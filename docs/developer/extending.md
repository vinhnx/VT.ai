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

## Extending Image Generation

VT.ai now uses GPT-Image-1 for image generation with enhanced capabilities and configuration options. Here's how to extend or customize the image generation functionality:

### Customizing Image Generation Settings

You can modify the default settings for GPT-Image-1 by updating the configuration in `vtai/utils/llm_providers_config.py`:

```python
# Example: Changing default image generation settings
DEFAULT_IMAGE_GEN_IMAGE_SIZE = "1536x1024"  # Default to landscape format
DEFAULT_IMAGE_GEN_BACKGROUND = "transparent"  # Default to transparent backgrounds
DEFAULT_IMAGE_GEN_OUTPUT_FORMAT = "png"  # Default to PNG format
DEFAULT_IMAGE_GEN_MODERATION = "auto"  # Default moderation level
DEFAULT_IMAGE_GEN_OUTPUT_COMPRESSION = 90  # Higher quality default
```

### Extending Image Processing Features

The image generation process is handled in `vtai/utils/media_processors.py`. To extend this functionality:

1. **Add New Image Processing Features**

   You can add post-processing features for generated images:

   ```python
   async def process_generated_image(image_data, format="jpeg"):
       """
       Apply custom processing to generated images.

       Args:
           image_data: Raw image data
           format: Image format (jpeg, png, webp)

       Returns:
           Processed image data
       """
       from PIL import Image, ImageFilter
       import io

       # Convert bytes to PIL Image
       image = Image.open(io.BytesIO(image_data))

       # Apply custom processing
       image = image.filter(ImageFilter.SHARPEN)

       # Convert back to bytes
       buffer = io.BytesIO()
       image.save(buffer, format=format.upper())

       return buffer.getvalue()
   ```

2. **Modify Image Generation UI**

   To add custom UI elements for your new image settings, update the settings in `vtai/utils/settings_builder.py`:

   ```python
   # Add a new setting for image generation
   settings_components.append(
       cl.Select(
           id="my_custom_image_setting",
           label="🖼️ My Custom Setting",
           values=["option1", "option2", "option3"],
           initial_value="option1"
       )
   )
   ```

3. **Add Custom Image Metadata**

   You can modify how image metadata is displayed by updating the `handle_trigger_async_image_gen` function:

   ```python
   # Add custom metadata to the image display
   metadata = {
       # ...existing metadata...
       "Custom Info": "Your custom information",
       "Processing": f"Applied {your_custom_process}"
   }
   ```

### Saving and Managing Generated Images

Generated images are now saved in an `imgs` directory with timestamped filenames. To extend this functionality:

1. **Custom Storage Locations**

   You can modify where images are stored:

   ```python
   # Example: Change image storage location or naming convention
   timestamp = int(time.time())
   img_dir = Path("custom_images_folder")
   img_dir.mkdir(exist_ok=True)
   img_path = img_dir / f"custom_prefix_{timestamp}.{format}"
   ```

2. **Image Organization Features**

   You could add features to organize images by prompt or category:

   ```python
   # Organize images by category derived from prompt
   def get_category_from_prompt(prompt):
       # Simple category extraction
       if "landscape" in prompt.lower():
           return "landscapes"
       elif "portrait" in prompt.lower():
           return "portraits"
       return "miscellaneous"

   category = get_category_from_prompt(query)
   img_dir = Path(f"imgs/{category}")
   img_dir.mkdir(exist_ok=True, parents=True)
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
