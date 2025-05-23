---
applyTo: '**'
---
# VT.ai Development Guide

## Commands
- **Run app**: `chainlit run vtai/app.py` or `python -m vtai.app`
- **Tests**: `python -m pytest` (all), `python -m pytest tests/unit/test_file.py::test_function` (single test)
- **Lint**: `ruff check vtai/` and `ruff format vtai/`
- **Install deps**: `uv pip install -U package_name`

## Code Style
- Use **tabs** for indentation (Copilot rules override)
- snake_case for variables/functions, PascalCase for classes
- Line limit: 100 characters (ruff configured)
- Type hints required for function signatures
- Google-style docstrings for public functions/classes

## Imports
Standard → third-party → local imports. Use lazy imports for performance-critical paths.

## Error Handling
- Specific exceptions with user-friendly messages
- Context managers for async operations
- Graceful degradation for optional features
- Log errors with `logger.error(f"Error: {type(e).__name__}: {str(e)}")`

## Project Structure
- `vtai/`: Main package (app.py is entry point)
- `tests/`: pytest structure mirroring src
- Use `uv` for all pip operations

## MCP



MCP tools: you can use avaiable mcp tools and web search mcp tool if needed. For example, if you need to search for a specific library or tool, you can use the web search mcp tool. If you need to check the status of a service or API, you can use the available mcp tools. You can also use the web search mcp tool to find documentation or examples for specific libraries or tools. For git mcp tool, you can use it to check the status of your git repository, create branches, or commit changes. You can browse avaiable mcp tools to see what is available inside .settings.json.
