"""
Configuration utilities for the VT application.

Manages application initialization, logging, and environment configuration.
"""

import importlib.resources
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Set TOKENIZERS_PARALLELISM explicitly at module level before any imports
# This prevents the HuggingFace tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import dotenv
import httpx
import litellm
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

# Update imports to use vtai namespace
from router.constants import RouteLayer
from utils import constants as const

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vt.ai")

# Create temporary directory for TTS audio files
temp_dir = tempfile.TemporaryDirectory()

# List of allowed mime types
allowed_mime = [
    "text/csv",
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
]

# Cache for API keys to avoid repeated environment checks
_api_keys_cache = {}

# Cache for model prices data
_model_prices_cache = {}
_model_prices_cache_file = Path(
    os.path.expanduser("~/.config/vtai/model_prices_cache.json")
)
_model_prices_cache_expiry = 24 * 60 * 60  # 24 hours in seconds


def load_api_keys() -> None:
    """
    Load API keys from environment variables and set them in os.environ.
    Prioritizes user-specific .env file before falling back to project .env
    Logs which keys were successfully loaded to help with debugging.
    Uses caching to avoid redundant loading operations.
    """
    global _api_keys_cache

    # Skip if already cached and VT_FAST_START is enabled
    if _api_keys_cache and os.environ.get("VT_FAST_START") == "1":
        return

    # First try to load from user config directory
    user_config_dir = os.path.expanduser("~/.config/vtai")
    user_env_path = os.path.join(user_config_dir, ".env")

    env_loaded = False

    # Try user config first
    if os.path.exists(user_env_path):
        load_dotenv(dotenv_path=user_env_path, override=True)
        logger.info(f"Loaded API keys from user config: {user_env_path}")
        env_loaded = True

    # Fall back to project .env if user config not found or as additional source
    project_env_path = dotenv.find_dotenv()
    if project_env_path:
        load_dotenv(
            dotenv_path=project_env_path, override=False
        )  # Don't override user config
        if not env_loaded:
            logger.info(f"Loaded API keys from project .env: {project_env_path}")
            env_loaded = True

    # Get API keys from environment
    api_key_names = [
        "OPENAI_API_KEY",
        "COHERE_API_KEY",
        "HUGGINGFACE_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "MISTRAL_API_KEY",
        "DEEPSEEK_API_KEY",
        "TAVILY_API_KEY",
        "LM_STUDIO_API_BASE",
        "LM_STUDIO_API_KEY",
    ]

    # Set API keys in environment
    loaded_keys = []
    for key in api_key_names:
        value = os.getenv(key)
        if value:
            os.environ[key] = value
            _api_keys_cache[key] = value
            loaded_keys.append(key)

    if loaded_keys:
        logger.info(f"Loaded API keys: {', '.join(loaded_keys)}")
    else:
        logger.warning("No API keys were loaded from environment")


def create_openai_clients() -> Tuple[OpenAI, AsyncOpenAI]:
    """
    Create OpenAI clients with optimized connection settings.

    Returns:
        Tuple of (sync_client, async_client)
    """
    # Configure timeout settings for better connection handling
    timeout_settings = httpx.Timeout(
        connect=10.0,  # Connection timeout
        read=300.0,  # Read timeout for longer operations
        write=60.0,  # Write timeout
        pool=10.0,  # Connection pool timeout
    )

    # Create synchronous client with custom timeout
    sync_client = OpenAI(
        timeout=timeout_settings,
        max_retries=3,  # Increase retries to handle transient errors
        http_client=httpx.Client(timeout=timeout_settings),
    )

    # Create asynchronous client with custom timeout
    async_client = AsyncOpenAI(
        timeout=timeout_settings,
        max_retries=3,
        http_client=httpx.AsyncClient(
            timeout=timeout_settings,
            limits=httpx.Limits(
                max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0
            ),
        ),
    )

    return sync_client, async_client


def get_openai_client() -> OpenAI:
    """
    Returns a synchronous OpenAI client with optimized connection settings.

    This is a convenience function that returns the first element of the tuple
    returned by create_openai_clients().

    Returns:
        OpenAI: A configured OpenAI client
    """
    sync_client, _ = create_openai_clients()
    return sync_client


def load_model_prices() -> Dict[str, Any]:
    """
    Load model pricing data with caching to avoid network requests on every startup.

    Returns:
        Dict containing model pricing data
    """
    global _model_prices_cache

    # Fast path for startup
    if _model_prices_cache:
        return _model_prices_cache

    # Check for cached price data file
    cache_file = _model_prices_cache_file

    try:
        # Create cache directory if it doesn't exist
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Check if cache file exists and is recent
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime

            # Use cache if it's less than expiry time (default 24 hours)
            if cache_age < _model_prices_cache_expiry:
                with cache_file.open("r") as f:
                    _model_prices_cache = json.load(f)
                    logger.info(
                        f"Using cached model prices data (age: {cache_age:.1f} seconds)"
                    )
                    return _model_prices_cache

        # If we get here, either no cache or cache too old
        # Only fetch from network if not in fast startup mode
        if os.environ.get("VT_FAST_START") != "1":
            logger.info("Fetching fresh model pricing data")
            pricing_url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

            with httpx.Client(timeout=10.0) as client:
                response = client.get(pricing_url)
                response.raise_for_status()

                # Save to cache
                price_data = response.json()
                with cache_file.open("w") as f:
                    json.dump(price_data, f)

                _model_prices_cache = price_data
                return price_data
        else:
            # In fast startup mode, use old cache if available or return empty dict
            if _model_prices_cache:
                logger.info("Using stale model prices cache due to fast startup mode")
                return _model_prices_cache
            elif cache_file.exists():
                with cache_file.open("r") as f:
                    _model_prices_cache = json.load(f)
                    logger.info(
                        "Using stale model prices cache due to fast startup mode"
                    )
                    return _model_prices_cache
            else:
                logger.warning("No model prices cache available in fast startup mode")
                return {}

    except Exception as e:
        logger.warning(f"Error loading model prices: {e}")
        # Return empty dict on error
        return {}


def initialize_encoder(lazy_load: bool = False) -> Optional[Any]:
    """
    Initialize the FastEmbedEncoder for semantic routing.
    Can be lazily loaded to speed up startup.

    Args:
        lazy_load: If True, return None to defer loading until needed

    Returns:
        Encoder instance or None if lazy loading
    """
    # Skip encoder initialization in fast start mode if lazy_load is True
    if lazy_load and os.environ.get("VT_FAST_START") == "1":
        logger.info("Deferring encoder initialization due to fast startup mode")
        return None

    try:
        from semantic_router.encoders import FastEmbedEncoder

        # Set the default encoder explicitly to disable any potential fallback to OpenAIEncoder
        # Create the FastEmbedEncoder instance with explicit model specification
        model_name = "BAAI/bge-small-en-v1.5"
        logger.info(f"Initializing FastEmbedEncoder with model: {model_name}")
        encoder = FastEmbedEncoder(model_name=model_name)
        return encoder
    except Exception as e:
        logger.error(f"Failed to initialize encoder: {e}")
        return None


def initialize_app() -> Tuple[RouteLayer, str, OpenAI, AsyncOpenAI]:
    """
    Initialize the application configuration.

    Returns:
        Tuple of (route_layer, assistant_id, openai_client, async_openai_client)
    """
    start_time = time.time()
    fast_start = os.environ.get("VT_FAST_START") == "1"

    if fast_start:
        logger.info("Fast startup mode enabled")

    # Load API keys
    load_api_keys()

    # Model alias map for litellm
    litellm.model_alias_map = const.MODEL_ALIAS_MAP

    # Configure litellm for better timeout handling
    litellm.request_timeout = 60  # 60 seconds timeout

    # Preload model prices in background if in normal mode
    # or use cached prices in fast mode
    try:
        load_model_prices()
    except Exception as e:
        logger.warning(f"Error preloading model prices: {e}")

    # Load semantic router layer from JSON file
    import json

    from semantic_router import Route
    from semantic_router.encoders import FastEmbedEncoder

    # Initialize encoder - potentially lazily in fast mode
    encoder = initialize_encoder(lazy_load=fast_start)

    # Use a minimal route layer in fast start mode
    if fast_start and not encoder:
        # Create a minimal RouteLayer with empty routes
        # The real initialization will happen on first use
        from semantic_router import RouteLayer as EmptyRouteLayer

        route_layer = EmptyRouteLayer(routes=[])
        logger.info("Created minimal route layer for fast startup")
    else:
        # Normal initialization with encoder
        try:
            # First try to load from package resources (for pip installation)
            with (
                importlib.resources.files("vtai.router")
                .joinpath("layers.json")
                .open("r") as f
            ):
                router_json = json.load(f)

                # Create routes from the JSON data
                routes = []

                for route_data in router_json["routes"]:
                    route_name = route_data["name"]
                    route_utterances = route_data["utterances"]

                    # Create Route object - passing the required utterances field and our encoder
                    route = Route(
                        name=route_name,
                        utterances=route_utterances,
                        encoder=encoder,  # Pass the same encoder instance to each route
                    )
                    routes.append(route)

                # Create RouteLayer with the routes and encoder
                route_layer = RouteLayer(routes=routes, encoder=encoder)

        except (ImportError, FileNotFoundError) as e:
            # Fallback to original behavior for development
            logger.warning(f"Could not load layers.json from package resources: {e}")
            try:
                # Try the original path as last resort
                with open("./vtai/router/layers.json", "r") as f:
                    router_json = json.load(f)

                    # Create routes from the JSON data
                    routes = []
                    for route_data in router_json["routes"]:
                        route_name = route_data["name"]
                        route_utterances = route_data["utterances"]

                        # Create Route object - passing the required utterances field
                        route = Route(
                            name=route_name,
                            utterances=route_utterances,
                            encoder=encoder,  # Use the same encoder instance
                        )
                        routes.append(route)

                # Create RouteLayer with the routes and explicitly pass the encoder
                route_layer = RouteLayer(routes=routes, encoder=encoder)
            except Exception as e:
                logger.error(f"Failed to load routes: {e}")
                # Create empty route layer if all else fails
                route_layer = RouteLayer(
                    routes=[], encoder=encoder
                )  # Still provide the encoder

    # Get assistant ID
    assistant_id = os.environ.get("ASSISTANT_ID")

    # Initialize OpenAI clients
    openai_client, async_openai_client = create_openai_clients()

    end_time = time.time()
    logger.info(f"App initialization completed in {end_time - start_time:.2f} seconds")

    return route_layer, assistant_id, openai_client, async_openai_client
