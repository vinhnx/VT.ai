"""
Unit tests for VT.ai config utilities.
"""
import os
from unittest.mock import MagicMock, patch

from vtai.utils.config import initialize_app, load_api_keys


@patch("vtai.utils.config.dotenv.find_dotenv")
@patch("vtai.utils.config.load_dotenv")
def test_load_api_keys(mock_load_dotenv, mock_find_dotenv, monkeypatch):
    """Test that API keys loading works correctly."""
    # Setup mocks
    mock_find_dotenv.return_value = "/mock/path/.env"
    mock_load_dotenv.return_value = True

    # Mock environment variables
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")

    # Call the function
    load_api_keys()

    # Verify dotenv was loaded
    mock_load_dotenv.assert_called()

    # Verify API key was set in environment
    assert os.environ.get("OPENAI_API_KEY") == "test_api_key"


@patch("vtai.utils.config.load_api_keys")
@patch("vtai.utils.config.create_openai_clients")
@patch("json.load")  # Patch the correct json module directly
@patch("vtai.utils.config.open", create=True)  # Use create=True for functions that might not exist in all contexts
@patch("semantic_router.encoders.FastEmbedEncoder")
@patch("semantic_router.Route")
@patch("vtai.router.constants.RouteLayer")
def test_initialize_app(mock_route_layer, mock_route, mock_encoder, mock_open,
                      mock_json_load, mock_create_clients, mock_load_keys):
    """Test application initialization."""
    # Setup mocks
    mock_encoder_instance = MagicMock()
    mock_encoder.return_value = mock_encoder_instance

    mock_route_instance = MagicMock()
    mock_route.return_value = mock_route_instance

    mock_route_layer_instance = MagicMock()
    mock_route_layer.return_value = mock_route_layer_instance

    mock_openai_client = MagicMock()
    mock_async_openai_client = MagicMock()
    mock_create_clients.return_value = (mock_openai_client, mock_async_openai_client)

    # Mock JSON data
    mock_json_load.return_value = {
        "routes": [
            {
                "name": "test-route",
                "utterances": ["test utterance"]
            }
        ]
    }

    # Mock importlib.resources.files
    resources_mock = MagicMock()
    with patch("vtai.utils.config.importlib.resources.files", return_value=resources_mock):
        # Add enough mocking to make the test run
        mock_file = MagicMock()
        mock_open_result = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_open_result
        resources_mock.joinpath.return_value = mock_file

        # Simulate the importlib exception to reach the fallback code path
        resources_mock.joinpath.side_effect = ImportError("Test error")

        # Call the function
        result = initialize_app()

    # Verify results - should be a 4-tuple
    assert isinstance(result, tuple)
    assert len(result) == 4  # route_layer, assistant_id, openai_client, async_openai_client

    # Verify API keys were loaded
    mock_load_keys.assert_called_once()    mock_load_keys.assert_called_once()