<p align="center">
  <em>VT.ai - Multi-modalities LLMs chat application</em>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![GitHub User's stars](https://img.shields.io/github/stars/vinhnx)](https://github.com/vinhnx)
[![HackerNews User Karma](https://img.shields.io/hackernews/user-karma/vinhnx)](https://news.ycombinator.com/user?id=vinhnx)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/vinhnx)](https://x.com/vinhnx)

---

VT.ai is a Python application built using the Chainlit library, providing a chat interface for interacting with large language models (LLMs) from various providers. This application supports multi-modal conversations, allowing users to upload files and leverage LLMs for text, image, and vision processing.

---

### Multi LLM Providers Selection with LiteLLM

![Multi LLM Providers](./src/vtai/resources/screenshot/1.jpg)

### Multi-modal Conversation Routing with SemanticRouter

![Multi-modal Conversation](./src/vtai/resources/screenshot/2.jpg)

## Prerequisites

This project uses `rye` as the Python dependencies management tool. Before installing the dependencies, make sure you have `rye` installed:

Install `rye`

```
curl -sSf https://rye-up.com/get | bash
```

> Rye ships an env file which should be sourced to update PATH automatically.
>
> ```
> echo 'source "$HOME/.rye/env"' >> ~/.zprofile
> ```
>
> In some setups `.zprofile` is not sourced, in which case you can add it to your .zshrc:
>
> ```
> echo 'source "$HOME/.rye/env"' >> ~/.zshrc
> ```
>
> There is a quite a bit to shims and their behavior. Make sure to read up on shims to learn more.

Ref: https://rye-up.com/guide/installation/#add-shims-to-path

## Features

-   Select an LLM model from a list of available models (OpenAI, Anthropic, Google, and more)
-   Upload files (text, images) for multi-modal processing
-   Stream responses from the LLM in real-time
-   Update settings (model, temperature, top-p, etc.) during the chat session
-   Dynamic conversation routing using SemanticRouter for accurate modality selection
-   Multi-modal input/output integration (text, images)

## Installation

1. Clone the repository and rename it as `vtai` (optional): `git clone https://github.com/vinhnx/VT.ai.git vtai`
1. Navigate to the project directory: `cd vtai`
1. Install `rye` (Python packages manager), guide: https://github.com/astral-sh/rye?tab=readme-ov-file#installation
1. Start dependencies sync: `rye sync`
1. Activate Python virtual environment: `source .venv/bin/activate`
1. Run the app: `chainlit run src/vtai/app.py -w`

## Usage

1. Rename the `.env.example` file to `.env` and configure your private LLM provider API keys.
2. Set up the required configuration files (`config.py` and `llm_profile_builder.py`) with your LLM models and settings.
3. Run the app with optional hot reload: `chainlit run src/vtai/app.py -w`
4. Open the provided URL in your web browser (e.g., `localhost:8000`).
5. Select an LLM model from the available options.
6. Customize the LLM parameters (optional).
7. Start chatting with the LLM or upload files for multi-modal processing.

## Dependencies

This application relies on the following main dependencies:

-   `chainlit`: A library for building chat applications with language models.
-   `litellm`: A library for interacting with large language models.
-   `semantic-router`: A library for fast and accurate decision-making in conversation routing.

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b my-new-feature`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions, feel free to reach out:

-   Twitter: [@vinhnx](https://twitter.com/vinhnx)
