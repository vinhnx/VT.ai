## v0.7.5 (2026-04-02)

### Security Fixes

- **Dependency Security Updates**: Fixed 25 Dependabot alerts by updating vulnerable dependencies
  - aiohttp 3.13.3 → 3.13.5 (fix 5 CVEs: HTTP header injection, response splitting, DoS)
  - cryptography 46.0.4 → 46.0.6 (fix SECT curve subgroup validation bypass)
  - mcp 1.12.4 → 1.26.0 (fix DNS rebinding protection)
  - PyJWT 2.11.0 → 2.12.1 (fix critical header validation bypass)
  - black 26.1.0 → 26.3.1 (fix path traversal in cache files)
  - onnx 1.19.0 → 1.21.0 (fix TOCTOU arbitrary file read/write)
  - pillow 10.4.0 → 12.2.0 (fix PSD out-of-bounds write)

### Other Changes

- Pin Python version to 3.11.x to resolve dependency conflicts
- Add uv override-dependencies for transitive security fixes

## Upcoming (Next Release)

### Features

- **Web Search Summarization**: Added intelligent summarization capability to web search results that synthesizes information from multiple sources into coherent, readable answers while preserving source attribution
- **Configurable Search Experience**: Added new setting to toggle between summarized and raw search results based on user preference
- **Enhanced Search Documentation**: Updated user docs with comprehensive information about the web search summarization feature

## v0.1.15 (2025-04-10)

### Other Changes

- cff2e96 Remove unused icon mapping for gpt-4o-mini and format list comprehensions for better readability (Vinh Nguyen)
- 420bb59 Refactor README and routing logic: streamline content, remove unused routes, and enhance message handling in conversation handlers (Vinh Nguyen)
- cf02386 Update CHANGELOG.md to include recent enhancements in release automation and changelog generation (Vinh Nguyen)
- afdd172 Enhance release automation: add changelog generation and GitHub release creation; refactor version update process (Vinh Nguyen)

## v0.1.14 (2025-04-10)

### Other Changes

- afdd172 Enhance release automation: add changelog generation and GitHub release creation; refactor version update process (Vinh Nguyen)

## v0.1.14 (2025-04-10)

### Other Changes

- d4c84ad Refactor chat profile starters to shuffle on app startup; add random prompt generation functions for enhanced user experience (Vinh Nguyen)
- ae61d6c Enhance routing capabilities and add specialized prompts for conversation types; update README for clarity on supported routes and training instructions (Vinh Nguyen)
- 12c050f Fix typos in image generation starter message for clarity (Vinh Nguyen)
- e92b4ad Add development mode instructions to README; update LLM provider configuration and default settings (Vinh Nguyen)
- b112e56 Update README for application run instructions and upgrade methods; modify pyproject.toml target version to Python 3.12 (Vinh Nguyen)
- bc4ebaf Refactor LLM provider settings: rename module and update imports across conversation handlers, media processors, and settings builder (Vinh Nguyen)
- a61deac Refactor README for clarity on installation methods and voice interaction; update LLM model aliases in provider settings (Vinh Nguyen)
- b089d52 Update README to clarify voice interaction features and remove outdated interface controls (Vinh Nguyen)
- 87038f4 Update README from llm docs inspiration (Vinh Nguyen)
- 1bdd2ad Update README.md to include quick start instructions with uvx and upgrade methods (Vinh Nguyen)
