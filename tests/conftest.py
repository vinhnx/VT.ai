"""
Pytest configuration file for VT.ai tests.

This file contains fixtures and configurations used across test files.
"""

import os
import sys

import pytest

# Add the project root to the Python path to allow imports from the vtai package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_config():
    """Fixture that returns a sample configuration for testing."""
    return {"api_key": "test_api_key", "model": "test-model", "temperature": 0.7}


@pytest.fixture
def mock_llm_response():
    """Fixture that returns a mock LLM response."""
    return {
        "id": "mock-response-id",
        "object": "chat.completion",
        "created": 1713894327,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response from the LLM.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
