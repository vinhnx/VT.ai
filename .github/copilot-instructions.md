# Custom instructions for Copilot

## Project context

This project, VT.ai, appears to be focused on AI applications, potentially leveraging language models via tools like LiteLLM. The goal is to create robust and efficient AI-powered features or services. Key technologies include Python.

## Indentation

We use tabs, not spaces.

## Coding style

- Use snake_case for variables, functions, and filenames.
- Use PascalCase for class names.
- Add docstrings to all public modules, classes, functions, and methods using Google Python Style Guide.
- Keep lines under 120 characters.
- Use type hints for function signatures.

## Testing

- Use `pytest` for writing and running unit and integration tests.
- Place tests in the `tests/` directory, mirroring the structure of the `src/` directory.
- Aim for high test coverage for all new code.

## Dependencies

- Key dependencies include Python 3.x and potentially libraries like `litellm`, `requests`, etc. Manage dependencies using `pyproject.toml` and `uv`.
- Keep dependencies updated.

## File structure

- `src/`: Contains the main application source code.
  - `vt_ai/`: Main package directory.
- `tests/`: Contains all test files.
- `scripts/`: Utility scripts for development or deployment.
- `docs/`: Project documentation.
- `pyproject.toml`: Project metadata and dependencies.
- `.github/`: GitHub specific files like workflows and issue templates.

## API usage

- When interacting with external APIs (e.g., LLM providers), use API keys stored securely (e.g., environment variables or a secrets manager). Do not commit keys to the repository.
- Implement proper error handling and retry mechanisms for API calls.
- Be mindful of rate limits.

## Common pitfalls

- Avoid using mutable default arguments in function definitions.
- Handle exceptions gracefully, providing informative error messages.
- Ensure proper resource management (e.g., closing files or network connections).

## Security considerations

- Sanitize any external input to prevent injection attacks.
- Use environment variables or a dedicated secrets management solution for sensitive data like API keys and passwords.
- Regularly update dependencies to patch security vulnerabilities.

## Performance considerations

- Write efficient code and algorithms.
- Profile code to identify bottlenecks when necessary.
- Consider caching for frequently accessed data or expensive computations.

## Future development

- Future plans may include expanding model support, improving user interfaces, or adding new AI capabilities.

## Python workflow

Every pip command should be run with uv prefix, example: `uv pip install -U litellm`. In short, use uv or uvx
