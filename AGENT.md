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