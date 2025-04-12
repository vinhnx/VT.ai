"""
Integration tests for VT.ai router functionality.
"""
import pytest
from semantic_router import Route
from semantic_router.encoders import FastEmbedEncoder
from semantic_router.layer import RouteLayer


def test_router_initialization():
    """Test that the router can be initialized correctly."""
    try:
        # Create a test encoder
        encoder = FastEmbedEncoder(model_name="BAAI/bge-small-en-v1.5")

        # Create a test route
        test_route = Route(
            name="test-route",
            utterances=["Hello, how are you?", "What is the weather like today?"],
            encoder=encoder
        )

        # Create a router with the test route
        router = RouteLayer(routes=[test_route], encoder=encoder)

        # Verify router has expected properties
        assert router is not None

        # Test with a basic prompt
        test_prompt = "What is the weather like today?"
        # Call the __call__ method directly since that's what the RouteLayer uses
        route = router(test_prompt)

        # Verify route has expected structure
        assert route is not None
        assert hasattr(route, "name")
        assert isinstance(route.name, str)
    except Exception as e:
        pytest.fail(f"Router initialization failed with exception: {e}")        pytest.fail(f"Router initialization failed with exception: {e}")