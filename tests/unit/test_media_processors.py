"""
Unit tests for VT.ai media processors.
"""

import os
import tempfile
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vtai.utils.media_processors import handle_trigger_async_image_gen, handle_vision


class MockResponse:
    """Mock response object for litellm.acompletion"""

    def __init__(self, content):
        self.choices = [self.MockChoice(content)]

    class MockChoice:
        def __init__(self, content):
            self.message = self.MockMessage(content)

        class MockMessage:
            def __init__(self, content):
                self.content = content


# Mock the Chainlit context
@contextmanager
def mock_chainlit_context():
    """Context manager to mock Chainlit context."""
    mock_context = MagicMock()
    mock_context.session = MagicMock()

    with patch("chainlit.context.get_context", return_value=mock_context):
        # Mock other chainlit related calls
        with patch("chainlit.user_session", new=MagicMock()) as mock_user_session:
            # Setup basic session data
            chat_settings = {
                "settings_vision_model": "gpt-4-vision",
                "settings_image_gen_image_style": "vibrant",
                "settings_image_gen_image_quality": "hd",
            }
            mock_user_session.get.return_value = chat_settings
            yield mock_user_session


@pytest.mark.asyncio
@patch("vtai.utils.media_processors.cl.Message")
@patch("vtai.utils.media_processors.litellm.acompletion")
@patch("vtai.utils.media_processors.litellm.supports_vision")
async def test_handle_vision(mock_supports_vision, mock_acompletion, mock_cl_message):
    """Test handling of vision processing."""
    # Make sure supports_vision returns True
    mock_supports_vision.return_value = True

    # Setup mock for cl.Message
    mock_message_instance = MagicMock()
    mock_message_instance.send = AsyncMock()
    mock_cl_message.return_value = mock_message_instance

    # Create a mock response for acompletion
    mock_acompletion.return_value = MockResponse(
        "This is an image of a mountain landscape."
    )

    # Create temporary image file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        # Create a test image
        with open(temp_filename, "wb") as f:
            f.write(b"test image data")

        # Use our context manager to mock Chainlit context
        with mock_chainlit_context():
            # Additional mocks
            with patch(
                "vtai.utils.user_session_helper.get_setting",
                return_value="gpt-4-vision",
            ):
                with patch("vtai.utils.media_processors.cl.Image") as mock_cl_image:
                    with patch("vtai.utils.media_processors.cl.Text") as mock_cl_text:
                        with patch(
                            "vtai.utils.media_processors.update_message_history_from_assistant"
                        ) as mock_update_history:
                            with patch(
                                "vtai.utils.media_processors.asyncio.wait_for",
                                side_effect=lambda coro, timeout: coro,
                            ):
                                with patch(
                                    "vtai.utils.media_processors.get_user_session_id"
                                ) as mock_get_user_id:
                                    # Call the function
                                    await handle_vision(
                                        temp_filename, "Describe this image", True
                                    )

                                    # Verify acompletion was called
                                    mock_acompletion.assert_called_once()

                                    # Verify message instances were sent
                                    assert mock_message_instance.send.call_count >= 1
    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


@pytest.mark.asyncio
@patch("vtai.utils.media_processors.cl.Message")
@patch("vtai.utils.media_processors.litellm.aimage_generation")
async def test_handle_trigger_async_image_gen(mock_aimage_generation, mock_cl_message):
    """Test image generation."""
    # Mock the message instance
    mock_message_instance = MagicMock()
    mock_message_instance.send = AsyncMock()
    mock_cl_message.return_value = mock_message_instance

    # Mock the response from aimage_generation
    mock_response = {
        "data": [
            {
                "url": "https://example.com/image.jpg",
                "revised_prompt": "A beautiful mountain landscape",
            }
        ]
    }
    mock_aimage_generation.return_value = mock_response

    # Use our context manager to mock Chainlit context
    with mock_chainlit_context():
        # Additional mocks
        with patch("vtai.utils.user_session_helper.get_setting") as mock_get_setting:
            # Set up side_effect to return different values on each call
            mock_get_setting.side_effect = ["vibrant", "hd"]

            with patch(
                "vtai.utils.media_processors.get_user_session_id"
            ) as mock_get_user_id:
                with patch("vtai.utils.media_processors.cl.Image") as mock_cl_image:
                    with patch("vtai.utils.media_processors.cl.Text") as mock_cl_text:
                        with patch(
                            "vtai.utils.media_processors.cl.Action"
                        ) as mock_cl_action:
                            with patch(
                                "vtai.utils.media_processors.update_message_history_from_assistant"
                            ) as mock_update_history:
                                with patch(
                                    "vtai.utils.media_processors.asyncio.wait_for",
                                    side_effect=lambda coro, timeout: coro,
                                ):
                                    # Call the function
                                    await handle_trigger_async_image_gen(
                                        "Generate a mountain landscape"
                                    )

                                    # Verify aimage_generation was called
                                    mock_aimage_generation.assert_called_once()

                                    # Verify the message was sent
                                    assert (
                                        mock_message_instance.send.call_count >= 1
                                    )  # Verify the message was sent
                                    assert mock_message_instance.send.call_count >= 1
