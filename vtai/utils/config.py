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
from functools import lru_cache
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

# Configure logging - set level based on environment
log_level = os.environ.get("VT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
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

# Cache for router data
_router_cache = {}
_router_cache_file = Path(os.path.expanduser("~/.config/vtai/router_cache.json"))
_router_cache_expiry = 7 * 24 * 60 * 60  # 7 days in seconds

# Shared HTTP client for better connection pooling
_http_client = None
_async_http_client = None


def get_http_client() -> httpx.Client:
    """
    Get or create a shared HTTP client with optimized settings.

    Returns:
        A configured HTTP client
    """
    global _http_client

    if _http_client is None:
        timeout_settings = httpx.Timeout(
            connect=10.0,
            read=60.0,
            write=30.0,
            pool=10.0,
        )

        _http_client = httpx.Client(
            timeout=timeout_settings,
            follow_redirects=True,
            http2=False,  # Disable HTTP/2 to avoid dependency requirement
        )

    return _http_client


def get_async_http_client() -> httpx.AsyncClient:
    """
    Get or create a shared async HTTP client with optimized settings.

    Returns:
        A configured async HTTP client
    """
    global _async_http_client

    if _async_http_client is None:
        timeout_settings = httpx.Timeout(
            connect=10.0,
            read=60.0,
            write=30.0,
            pool=10.0,
        )

        _async_http_client = httpx.AsyncClient(
            timeout=timeout_settings,
            follow_redirects=True,
            http2=False,  # Disable HTTP/2 to avoid dependency requirement
            limits=httpx.Limits(
                max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0
            ),
        )

    return _async_http_client


def load_api_keys() -> None:
    """
    Load API keys from environment variables and set them in os.environ.
    Prioritizes user-specific .env file before falling back to project .env.
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


@lru_cache(maxsize=2)
def create_openai_clients() -> Tuple[OpenAI, AsyncOpenAI]:
    """
    Create OpenAI clients with optimized connection settings.
    Results are cached to avoid duplicate client creation.

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
        http_client=get_http_client(),
    )

    # Create asynchronous client with custom timeout
    async_client = AsyncOpenAI(
        timeout=timeout_settings,
        max_retries=3,
        http_client=get_async_http_client(),
    )

    return sync_client, async_client


def get_openai_client() -> OpenAI:
    """
    Returns a cached synchronous OpenAI client with optimized connection settings.

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

            client = get_http_client()
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


def cache_router_data(route_data: Dict[str, Any]) -> None:
    """
    Cache router data to disk to speed up future startups.

    Args:
        route_data: Router configuration data to cache
    """
    cache_file = _router_cache_file

    try:
        # Create cache directory if it doesn't exist
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to cache file
        with cache_file.open("w") as f:
            json.dump(route_data, f)

        logger.info(f"Saved router data to cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Error caching router data: {e}")


def load_cached_router_data() -> Optional[Dict[str, Any]]:
    """
    Load router data from cache if available and not expired.

    Returns:
        Dict containing router data or None if not available
    """
    cache_file = _router_cache_file

    try:
        # Check if cache file exists and is recent
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime

            # Use cache if it's less than expiry time
            if cache_age < _router_cache_expiry:
                with cache_file.open("r") as f:
                    data = json.load(f)
                    logger.info(
                        f"Using cached router data (age: {cache_age:.1f} seconds)"
                    )
                    return data
    except Exception as e:
        logger.warning(f"Error loading cached router data: {e}")

    return None


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


def load_routes(encoder=None, use_cache: bool = True) -> list:
    """
    Load routes from JSON file or cache with improved performance.

    Args:
        encoder: Optional encoder instance to use
        use_cache: Whether to use cached route data

    Returns:
        List of Route objects
    """
    fast_start = os.environ.get("VT_FAST_START") == "1"

    # Try to load from cache first if in fast mode
    if fast_start and use_cache:
        cached_data = load_cached_router_data()
        if cached_data:
            try:
                from semantic_router import Route

                routes = []
                for route_data in cached_data.get("routes", []):
                    route_name = route_data["name"]
                    route_utterances = route_data["utterances"]

                    # Create Route object - passing the required utterances field and encoder
                    route = Route(
                        name=route_name,
                        utterances=route_utterances,
                        encoder=encoder,
                    )
                    routes.append(route)

                return routes
            except Exception as e:
                logger.warning(f"Error loading routes from cache: {e}")

    # Load from file if cache not available or not in fast mode
    try:
        from semantic_router import Route

        # Try package resources first (for pip installation)
        try:
            with (
                importlib.resources.files("vtai.router")
                .joinpath("layers.json")
                .open("r") as f
            ):
                router_json = json.load(f)

                # Cache for future use
                if use_cache:
                    cache_router_data(router_json)
        except (ImportError, FileNotFoundError) as e:
            # Fallback to original path
            logger.warning(f"Could not load layers.json from package resources: {e}")
            with open("./vtai/router/layers.json", "r") as f:
                router_json = json.load(f)

                # Cache for future use
                if use_cache:
                    cache_router_data(router_json)

        # Create routes from the JSON data
        routes = []
        for route_data in router_json.get("routes", []):
            route_name = route_data["name"]
            route_utterances = route_data["utterances"]

            # Create Route object - passing the required utterances field and encoder
            route = Route(
                name=route_name,
                utterances=route_utterances,
                encoder=encoder,
            )
            routes.append(route)

        return routes
    except Exception as e:
        logger.error(f"Failed to load routes: {e}")
        return []


def initialize_app() -> Tuple[RouteLayer, None, OpenAI, AsyncOpenAI]:
    """
    Initialize the application configuration.

    Returns:
        Tuple of (route_layer, None, openai_client, async_openai_client)
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

    # Initialize encoder - potentially lazily in fast mode
    encoder = initialize_encoder(lazy_load=fast_start)

    # Load semantic router layer
    if fast_start and not encoder:
        # Create a minimal RouteLayer with empty routes in fast mode
        # The real initialization will happen on first use
        from semantic_router import RouteLayer as EmptyRouteLayer

        route_layer = EmptyRouteLayer(routes=[])
        logger.info("Created minimal route layer for fast startup")
    else:
        # Normal initialization with encoder
        from semantic_router import Route

        routes = load_routes(encoder=encoder, use_cache=True)

        # Create RouteLayer with the routes
        route_layer = RouteLayer(routes=routes, encoder=encoder)
        logger.info(f"Initialized route layer with {len(routes)} routes")

    # Initialize OpenAI clients
    openai_client, async_openai_client = create_openai_clients()

    # Preload model prices in background if in normal mode or use cached prices in fast mode
    if not fast_start:
        try:
            load_model_prices()
        except Exception as e:
            logger.warning(f"Error preloading model prices: {e}")

    end_time = time.time()
    logger.info(f"App initialization completed in {end_time - start_time:.2f} seconds")

    return route_layer, None, openai_client, async_openai_client


def cleanup():
    """
    Clean up resources when the application is shutting down.
    """
    global _http_client, _async_http_client

    # Close HTTP clients
    if _http_client:
        _http_client.close()

    if _async_http_client:
        # Note: In real usage, you should await the close() call
        # We're providing a synchronous version for simplicity
        try:
            import asyncio

            asyncio.run(_async_http_client.aclose())
        except Exception:
            pass

    # Clean up temporary directory
    try:
        temp_dir.cleanup()
    except Exception as e:
        logger.warning(f"Error cleaning up temporary directory: {e}")
