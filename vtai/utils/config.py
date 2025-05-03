"""
Configuration utilities for the VT application.

Manages application initialization, logging, and environment configuration.
"""

import importlib.resources
import logging
import os
import tempfile
from typing import Tuple

# Set TOKENIZERS_PARALLELISM explicitly at module level before any imports
# This prevents the HuggingFace tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import dotenv
import httpx
import litellm
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

# Update imports to use vtai namespace
from vtai.router.constants import RouteLayer
from vtai.utils import constants as const

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


def load_api_keys() -> None:
    """
    Load API keys from environment variables and set them in os.environ.
    Prioritizes user-specific .env file before falling back to project .env
    Logs which keys were successfully loaded to help with debugging.
    """
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
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
        "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        "LM_STUDIO_API_BASE": os.getenv("LM_STUDIO_API_BASE"),
        "LM_STUDIO_API_KEY": os.getenv("LM_STUDIO_API_KEY"),
    }

    # Set API keys in environment
    loaded_keys = []
    for key, value in api_keys.items():
        if value:
            os.environ[key] = value
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


def initialize_app() -> Tuple[RouteLayer, str, OpenAI, AsyncOpenAI]:
    """
    Initialize the application configuration.

    Returns:
        Tuple of (route_layer, assistant_id, openai_client, async_openai_client)
    """
    # Load API keys
    load_api_keys()

    # Model alias map for litellm
    litellm.model_alias_map = const.MODEL_ALIAS_MAP

    # Configure litellm for better timeout handling
    litellm.request_timeout = 60  # 60 seconds timeout

    # Load semantic router layer from JSON file - use proper path for installed package
    import json

    from semantic_router import Route
    from semantic_router.encoders import FastEmbedEncoder

    # Set the default encoder explicitly to disable any potential fallback to OpenAIEncoder
    # Create the FastEmbedEncoder instance with explicit model specification
    model_name = "BAAI/bge-small-en-v1.5"
    encoder = FastEmbedEncoder(model_name=model_name)

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

            # Load routes directly from the original path
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

    return route_layer, assistant_id, openai_client, async_openai_client
