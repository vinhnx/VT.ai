<p align="center">
  <img src="./public/logo_dark.png" height="200" alt="icon" />
</p>

<h3 align="center">VT.ai</h3>

<p align="center">
  <em>Multi-modal AI Assistant</em>
</p>

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/vinhnx?style=social)](https://github.com/vinhnx)
[![HackerNews Karma](https://img.shields.io/hackernews/user-karma/vinhnx?style=social)](https://news.ycombinator.com/user?id=vinhnx)
[![Twitter Follow](https://img.shields.io/twitter/follow/vinhnx?style=social)](https://twitter.com/vinhnx)
[![Twitter Follow](https://img.shields.io/twitter/follow/vtdotai?style=social)](https://twitter.com/vtdotai)

## Introduction

VT.ai is a multi-modal AI Chatbot Assistant, offering a chat interface to interact with Large Language Models (LLMs) from various providers. Both via remote API or running locally with [Ollama](https://github.com/ollama/ollama/blob/main/README.md#quickstart).

The application supports multi-modal conversations, seamlessly integrating text, images, and vision processing with LLMs.

[Beta] Multi-modal AI Assistant support via OpenAI's Assistant API function calling.

---

## Key Features

- **[Beta] Assistant support:** Enjoy the assistance of Multi-modal AI Assistant through OpenAI's Assistant API. It can write and run code to answer math questions.
- **Multi-Provider Support:** Choose from a variety of LLM providers including OpenAI, Anthropic, and Google, with more to come.
- **Multi-Modal Conversations:** Experience rich, multi-modal interactions by uploading text and image files. You can even drag and drop images for the model to analyze.
- **Real-time Responses:** Stream responses from the LLM as they are generated.
- **Dynamic Settings:** Customize model parameters such as temperature and top-p during your chat session.
- **Clean and Fast Interface:** Built using Chainlit, ensuring a smooth and intuitive user experience.
- **Advanced Conversation Routing:** Utilizes SemanticRouter for accurate and efficient modality selection.

---

![Multi LLM Providers](./src/resources/screenshot/1.jpg)

![Multi-modal Conversation](./src/resources/screenshot/2.jpg)

---

## Getting Started

### Prerequisites

- Python 3.7 or higher
- (Optional -- Recommended) `rye` as the Python dependencies manager (installation guide below)
- For using local models with [Ollama](https://github.com/ollama/ollama/blob/main/README.md#quickstart):
  - Download the Ollama client from https://ollama.com/download
  - Download the desired Ollama models from https://ollama.com/library (e.g., `ollama pull llama3`)
  - Follow the Ollama installation and setup instructions

### Usage

1. Rename the `.env.example` file to `.env` and configure your desired LLM provider API keys. If using Ollama, you can leave the API keys blank.
2. Create a new virtual environment with Python 3: `python3 -m venv .venv`
3. Activate the virtual environment: `source .venv/bin/activate`
4. Install the requirements using pip3: `pip3 install -r requirements.txt`
6. (Optional) Run semantic trainer once. `python3 src/router/trainer.py`
7. Run the app with optional hot reload: `chainlit run src/app.py -w`
8. Open the provided URL in your web browser (e.g., `localhost:8000`).
9. Select an LLM model and start chatting or uploading files for multi-modal processing. If using Ollama, select the `Ollama` option from the model dropdown.
10. To run Ollama server for serving local LLM models, you can use the following commands:
  - Example to use Meta's Llama 3 model locally from Ollama: `ollama pull llama3` to download the `llama3` model (replace with the desired model name)
  - `ollama serve` to start the Ollama server
  - `ollama --help` for more options and details

## Technical Overview

### Dependencies

- [Chainlit](https://github.com/Chainlit/chainlit): A powerful library for building chat applications with LLMs, providing a clean and fast front-end.
- [LiteLLM](https://github.com/BerriAI/litellm): A versatile library for interacting with LLMs, abstracting away the complexities of different providers.
- [SemanticRouter](https://github.com/aurelio-labs/semantic-router): A high-performance library for accurate conversation routing, enabling dynamic modality selection.

### Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a new branch: `git checkout -b my-new-feature`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request

### Releases

See [releases tags](https://github.com/vinhnx/VT.ai/releases)

## License

This project is licensed under the MIT License.

## Contact

For questions, suggestions, or feedback, feel free to reach out:

- Twitter: [@vinhnx](https://twitter.com/vinhnx)
