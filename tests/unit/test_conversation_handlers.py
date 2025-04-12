"""
Unit tests for VT.ai conversation handlers.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vtai.utils.conversation_handlers import handle_conversation


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
            mock_user_session.get.return_value = {"settings_chat_model": "gpt-4"}
            yield mock_user_session


@pytest.mark.asyncio
@patch("vtai.utils.conversation_handlers.cl.Message")
@patch("vtai.utils.conversation_handlers.handle_trigger_async_chat")
async def test_handle_conversation(mock_handle_trigger, mock_cl_message):
    """Test handling of conversations."""
    # Mock the message instance for the response
    mock_message_instance = MagicMock()
    mock_message_instance.send = AsyncMock()
    mock_cl_message.return_value = mock_message_instance

    # Mock the trigger function
    mock_handle_trigger.return_value = None

    # Mock message and route layer
    mock_message = MagicMock()
    mock_message.content = "Hello, how are you?"

    mock_messages = [{"role": "system", "content": "You are a helpful assistant"}]
    mock_route_layer = None

    # Use our context manager to mock Chainlit context
    with mock_chainlit_context():
        # Additional mocks
        with patch("vtai.utils.user_session_helper.get_setting", return_value="gpt-4"):
            with patch(
                "vtai.utils.conversation_handlers.update_message_history_from_user"
            ) as mock_update_history:
                # Call the function
                await handle_conversation(mock_message, mock_messages, mock_route_layer)

                # Verify the message was updated
                mock_update_history.assert_called_once_with("Hello, how are you?")

                # Verify the message was sent
                mock_message_instance.send.assert_called_once()

                # Verify the async chat handler was called
                mock_handle_trigger.assert_called_once()  # Verify the async chat handler was called
                mock_handle_trigger.assert_called_once()
