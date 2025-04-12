# Troubleshooting

This page provides solutions to common issues you might encounter when using VT.ai.

## API Key Issues

### Missing API Keys

**Problem**: VT.ai reports that API keys are missing or invalid.

**Solution**:

1. Check that you've set your API keys using one of these methods:

   ```bash
   # Command line
   vtai --api-key openai=sk-your-key

   # Environment variables
   export OPENAI_API_KEY=sk-your-key
   ```

2. Verify that the API keys are valid by testing them directly with the provider
3. Check the `~/.config/vtai/.env` file to ensure keys are saved correctly

### Rate Limiting

**Problem**: You're encountering rate limit errors from API providers.

**Solution**:

1. Switch to a different model or provider
2. Wait a few minutes before trying again
3. Check if your API key has usage restrictions
4. Consider upgrading your API tier with the provider

## Installation Problems

### Package Conflicts

**Problem**: Conflicts with existing Python packages during installation.

**Solution**:

1. Use a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install vtai
   ```

2. Try installing with uv instead:

   ```bash
   uv tool install vtai
   ```

3. Check for conflicting packages with `pip list`

### Missing Dependencies

**Problem**: VT.ai reports missing dependencies.

**Solution**:

1. Reinstall with all dependencies:

   ```bash
   pip install vtai[all]
   ```

2. Check if you have enough disk space
3. Try updating pip: `pip install --upgrade pip`

## Performance Issues

### Slow Responses

**Problem**: VT.ai is taking a long time to respond.

**Solution**:

1. Switch to a faster model (e.g., GPT-o3 Mini, Claude 3 Haiku)
2. Check your internet connection
3. Reduce query complexity
4. Disable dynamic routing if it's causing delays

### High Memory Usage

**Problem**: VT.ai is using excessive system memory.

**Solution**:

1. Restart the application
2. Use smaller models
3. Close other memory-intensive applications
4. Increase swap space if possible

## Feature-Specific Issues

### Image Generation Not Working

**Problem**: Image generation commands don't create images.

**Solution**:

1. Ensure you have a valid OpenAI API key with DALL-E access
2. Try more explicit prompts like "Generate an image of..."
3. Check for error messages in the console output
4. Verify that you have sufficient API credits for image generation

### TTS Not Working

**Problem**: Text-to-speech functionality isn't working.

**Solution**:

1. Make sure TTS is enabled in settings
2. Select a different TTS model
3. Check audio output settings on your device
4. Verify that you have the necessary API access for TTS features

### Vision Analysis Issues

**Problem**: Image analysis doesn't work properly.

**Solution**:

1. Ensure you're using a vision-capable model like GPT-4o
2. Check that the image format is supported (JPG, PNG, WebP)
3. Try with a different image
4. Make sure your query specifically references the image

## Chainlit Interface Issues

### UI Not Loading

**Problem**: The web interface doesn't load properly.

**Solution**:

1. Check if the Chainlit server is running (look for log messages)
2. Try a different web browser
3. Clear your browser cache
4. Check for port conflicts (default is 8000)

### Settings Not Saving

**Problem**: Settings changes don't persist between sessions.

**Solution**:

1. Check that you have write permission to `~/.config/vtai/`
2. Manually edit settings in the configuration files
3. Try running with elevated permissions if needed

## Getting More Help

If you're still experiencing issues:

1. Check the [GitHub repository](https://github.com/vinhnx/VT.ai) for open issues
2. Create a new issue with detailed information about your problem
3. Join the community discussions for assistance

*This page is under construction. More troubleshooting information will be added soon.*
