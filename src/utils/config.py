"""
Configuration module for VT.ai application.

Handles loading environment variables, API keys, and application configuration.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional

import dotenv
import litellm

from utils import llm_settings_config as conf
from semantic_router.layer import RouteLayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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
    "audio/wav"
]

def load_api_keys() -> None:
    """
    Load API keys from environment variables and set them in os.environ.
    Logs which keys were successfully loaded to help with debugging.
    """
    # Load .env file
    dotenv.load_dotenv(dotenv.find_dotenv())
    
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
        "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
    }
    
    # Set API keys in environment
    loaded_keys = []
    for key, value in api_keys.items():
        if value:
            os.environ[key] = value
            loaded_keys.append(key)
    
    logger.info(f"Loaded API keys: {', '.join(loaded_keys)}")
    if not loaded_keys:
        logger.warning("No API keys were loaded from environment")

def initialize_app():
    """
    Initialize the application configuration.
    
    Returns:
        Tuple of (route_layer, assistant_id)
    """
    # Load API keys
    load_api_keys()
    
    # Model alias map for litellm
    litellm.model_alias_map = conf.MODEL_ALIAS_MAP
    
    # Load semantic router layer from JSON file
    route_layer = RouteLayer.from_json("./src/router/layers.json")
    
    # Get assistant ID
    assistant_id = os.environ.get("ASSISTANT_ID")
    
    return route_layer, assistant_id