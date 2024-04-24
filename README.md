<p align="center">
  <img src="./public/logo_dark.png" height="200" alt="icon" />
</p>

<h3 align="center">VT.ai</h3>

<p align="center">
  <em>Multi-modal Large Language Models Chat Application</em>
</p>

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/vinhnx?style=social)](https://github.com/vinhnx)
[![HackerNews Karma](https://img.shields.io/hackernews/user-karma/vinhnx?style=social)](https://news.ycombinator.com/user?id=vinhnx)
[![Twitter Follow](https://img.shields.io/twitter/follow/vinhnx?style=social)](https://twitter.com/vinhnx)
[![Twitter Follow](https://img.shields.io/twitter/follow/vtdotai?style=social)](https://twitter.com/vtdotai)

## Introduction

VT.ai is a Python application that provides a user-friendly chat interface for interacting with Large Language Models (LLMs) from various providers. The application supports multi-modal conversations, allowing users to seamlessly integrate text, images, and vision processing in their interactions with the LLMs.

### Vision aware

![Multi-modal Conversation](./src/vtai/resources/screenshot/3.jpg)

### Multi LLM Providers Selection with LiteLLM

![Multi LLM Providers](./src/vtai/resources/screenshot/1.jpg)

### Multi-modal Conversation Routing with SemanticRouter

![Multi-modal Conversation](./src/vtai/resources/screenshot/2.jpg)

---

## Key Features

-   **Multi-Provider Support:** Choose from a list of LLM providers, including OpenAI, Anthropic, and Google, with more to come.
-   **Multi-Modal Conversations:** Upload text and image files for a rich, multi-modal experience. You can drag and image and have the model explain it for you.
-   **Real-time Responses:** Stream responses from the LLM as they are generated.
-   **Dynamic Settings:** Various settings to choose from. TODO: Adjust model parameters such as temperature and top-p during the chat session.
-   **Clean and Fast Interface:** Built using Chainlit, ensuring a smooth and intuitive user experience.
-   **Advanced Conversation Routing:** Leveraging SemanticRouter for accurate and efficient modality selection.

## Getting Started

### Prerequisites

-   Python 3.7 or higher
-   `rye` as the Python dependencies manager (installation guide below)

### Installation

1. Clone the repository: `git clone https://github.com/vinhnx/VT.ai.git vtai` (optional: rename the cloned directory to `vtai`)
2. Navigate to the project directory: `cd vtai`

If you had installed `rye` from [Prerequisites step](https://github.com/vinhnx/VT.ai?tab=readme-ov-file#prerequisites), you can skip these steps, process to [Usage](https://github.com/vinhnx/VT.ai?tab=readme-ov-file#usage) section below

> 1. Install `rye` (Python packages manager):
>     ```
>     curl -sSf https://rye-up.com/get | bash
>     ```
> 2. Source the Rye env file to update PATH (add this to your shell configuration file, e.g., `.zprofile` or `.zshrc`):
>     ```
>     source "$HOME/.rye/env"
>     ```

### Usage

1. Rename the `.env.example` file to `.env` and configure your private LLM provider API keys.
2. Start dependencies sync: `rye sync`
3. Activate the Python virtual environment: `source .venv/bin/activate`
4. Run the app with optional hot reload: `chainlit run src/vtai/app.py -w`
5. Open the provided URL in your web browser (e.g., `localhost:8000`).
6. Select an LLM model and start chatting or uploading files for multi-modal processing.

## Technical Overview

### Dependencies

-   [Chainlit](https://github.com/Chainlit/chainlit): A powerful library for building chat applications with LLMs, providing a clean and fast front-end.
-   [LiteLLM](https://github.com/BerriAI/litellm): A versatile library for interacting with LLMs, abstracting away the complexities of different providers.
-   [SemanticRouter](https://github.com/aurelio-labs/semantic-router): A high-performance library for accurate conversation routing, enabling dynamic modality selection.

### Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a new branch: `git checkout -b my-new-feature`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions, suggestions, or feedback, feel free to reach out:

-   Twitter: [@vinhnx](https://twitter.com/vinhnx)
