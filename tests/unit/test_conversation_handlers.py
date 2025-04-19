"""
Unit tests for VT.ai conversation handlers.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, call, patch

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


@pytest.mark.asyncio
@patch("vtai.utils.conversation_handlers.cl.Step")
@patch("vtai.utils.conversation_handlers.cl.Message")
@patch("vtai.utils.conversation_handlers.WebSearchTool")
async def test_handle_web_search_with_summarization(
    mock_web_search_tool, mock_cl_message, mock_cl_step
):
    """Test web search handling with summarization enabled."""
    # Mock the message instance
    mock_message_instance = MagicMock()
    mock_message_instance.content = ""
    mock_message_instance.stream_token = AsyncMock()
    mock_message_instance.update = AsyncMock()
    mock_cl_message.return_value = mock_message_instance

    # Mock the step instance for the search visualization
    mock_step_instance = AsyncMock()
    mock_step_instance.__aenter__.return_value = mock_step_instance
    mock_step_instance.__aexit__.return_value = None
    mock_step_instance.stream_token = AsyncMock()
    mock_step_instance.update = AsyncMock()
    mock_step_instance.remove = AsyncMock()
    mock_cl_step.return_value = mock_step_instance

    # Mock the WebSearchTool instance
    mock_web_search_instance = MagicMock()
    mock_web_search_instance.search = AsyncMock()
    mock_web_search_tool.return_value = mock_web_search_instance

    # Mock search result with summarized content
    mock_search_result = {
        "status": "success",
        "response": "This is a summarized response about quantum computing.",
        "sources_json": '{"sources": [{"title": "Quantum Computing Advances", "url": "https://example.com/quantum1"}, {"title": "Recent Research", "url": "https://example.com/quantum2"}]}',
        "model": "gpt-4o",
    }
    mock_web_search_instance.search.return_value = mock_search_result

    # Test query
    query = "What are the latest developments in quantum computing?"

    # Use our context manager to mock Chainlit context
    with mock_chainlit_context():
        # Setup the mocks for the settings and environment
        with patch(
            "vtai.utils.user_session_helper.get_setting",
            side_effect=lambda key: {
                "SETTINGS_SUMMARIZE_SEARCH_RESULTS": True,  # Summarization enabled
                "settings_chat_model": "gpt-4o",
            }.get(key, None),
        ):
            with patch(
                "os.environ.get",
                side_effect=lambda key, default=None: {
                    "OPENAI_API_KEY": "sk-test-key",
                    "TAVILY_API_KEY": None,
                }.get(key, default),
            ):
                with patch(
                    "vtai.utils.conversation_handlers.update_message_history_from_assistant"
                ):
                    with patch(
                        "vtai.utils.conversation_handlers.create_message_actions",
                        return_value=[],
                    ):
                        # Call the web search handler
                        from vtai.utils.conversation_handlers import handle_web_search

                        await handle_web_search(query, mock_message_instance)

    # Verify the WebSearchTool was created and called with correct parameters
    mock_web_search_tool.assert_called_once_with(
        api_key="sk-test-key", tavily_api_key=None
    )

    # Verify the search was performed with summarization enabled
    _, kwargs = mock_web_search_instance.search.call_args
    search_params = (
        kwargs.get("params", None) or mock_web_search_instance.search.call_args[0][0]
    )
    assert search_params.search_options.summarize_results is True

    # Verify the appropriate message prefix was used for summarized results
    prefix_call_args = [
        call[0][0]
        for call in mock_message_instance.stream_token.call_args_list
        if "Here's a summary" in call[0][0]
    ]
    assert (
        len(prefix_call_args) > 0
    ), "Did not find the summarization prefix in the response"


@pytest.mark.asyncio
@patch("vtai.utils.conversation_handlers.cl.Step")
@patch("vtai.utils.conversation_handlers.cl.Message")
@patch("vtai.utils.conversation_handlers.WebSearchTool")
async def test_handle_web_search_without_summarization(
    mock_web_search_tool, mock_cl_message, mock_cl_step
):
    """Test web search handling with summarization disabled."""
    # Mock the message instance
    mock_message_instance = MagicMock()
    mock_message_instance.content = ""
    mock_message_instance.stream_token = AsyncMock()
    mock_message_instance.update = AsyncMock()
    mock_cl_message.return_value = mock_message_instance

    # Mock the step instance for the search visualization
    mock_step_instance = AsyncMock()
    mock_step_instance.__aenter__.return_value = mock_step_instance
    mock_step_instance.__aexit__.return_value = None
    mock_step_instance.stream_token = AsyncMock()
    mock_step_instance.update = AsyncMock()
    mock_step_instance.remove = AsyncMock()
    mock_cl_step.return_value = mock_step_instance

    # Mock the WebSearchTool instance
    mock_web_search_instance = MagicMock()
    mock_web_search_instance.search = AsyncMock()
    mock_web_search_tool.return_value = mock_web_search_instance

    # Mock search result with raw content
    mock_search_result = {
        "status": "success",
        "response": "Raw search result about quantum computing from multiple sources.",
        "sources_json": '{"sources": [{"title": "Quantum Computing Advances", "url": "https://example.com/quantum1"}, {"title": "Recent Research", "url": "https://example.com/quantum2"}]}',
        "model": "gpt-4o",
    }
    mock_web_search_instance.search.return_value = mock_search_result

    # Test query
    query = "What are the latest developments in quantum computing?"

    # Use our context manager to mock Chainlit context
    with mock_chainlit_context():
        # Setup the mocks for the settings and environment
        with patch(
            "vtai.utils.user_session_helper.get_setting",
            side_effect=lambda key: {
                "SETTINGS_SUMMARIZE_SEARCH_RESULTS": False,  # Summarization disabled
                "settings_chat_model": "gpt-4o",
            }.get(key, None),
        ):
            with patch(
                "os.environ.get",
                side_effect=lambda key, default=None: {
                    "OPENAI_API_KEY": "sk-test-key",
                    "TAVILY_API_KEY": None,
                }.get(key, default),
            ):
                with patch(
                    "vtai.utils.conversation_handlers.update_message_history_from_assistant"
                ):
                    with patch(
                        "vtai.utils.conversation_handlers.create_message_actions",
                        return_value=[],
                    ):
                        # Call the web search handler
                        from vtai.utils.conversation_handlers import handle_web_search

                        await handle_web_search(query, mock_message_instance)

    # Verify the WebSearchTool was created and called with correct parameters
    mock_web_search_tool.assert_called_once_with(
        api_key="sk-test-key", tavily_api_key=None
    )

    # Verify the search was performed with summarization disabled
    _, kwargs = mock_web_search_instance.search.call_args
    search_params = (
        kwargs.get("params", None) or mock_web_search_instance.search.call_args[0][0]
    )
    assert search_params.search_options.summarize_results is False

    # Verify the appropriate message prefix was used for raw results
    prefix_call_args = [
        call[0][0]
        for call in mock_message_instance.stream_token.call_args_list
        if "Here's what I found" in call[0][0]
    ]
    assert (
        len(prefix_call_args) > 0
    ), "Did not find the raw results prefix in the response"
