"""
Constants for the semantic router component.
"""

from typing import List

from semantic_router import RouteLayer as SRRouteLayer

# Export RouteLayer directly to fix import issues
RouteLayer = SRRouteLayer

"""
Semantic router type definitions for VT.ai.

This module defines the semantic routing types used by the router layer to
classify and direct user queries to appropriate processing pathways.

The router uses semantic vector space to make fast routing decisions without
requiring LLM inference, improving response time and efficiency.
"""

from enum import Enum
from typing import Final, Set


class SemanticRouterType(str, Enum):
    """
    Enumeration of semantic router types for conversation classification.

    These types correspond to the route names defined in the `layers.json` file
    and are used to categorize user queries based on their semantic meaning.

    Each value represents a distinct conversation type that may require
    different processing or model selection.
    """

    IMAGE_GENERATION: Final[str] = "image-generation"
    TEXT_PROCESSING: Final[str] = "text-processing"
    CASUAL_CONVERSATION: Final[str] = "casual-conversation"
    CURIOUS: Final[str] = "curious"
    VISION_IMAGE_PROCESSING: Final[str] = "vision-image-processing"
    WEB_SEARCH: Final[str] = "web-search"

    @classmethod
    def values(cls) -> List[str]:
        """Get a list of all route type values."""
        return [item.value for item in cls]

    @classmethod
    def requires_image_processing(cls) -> Set[str]:
        """Get the set of routes that require image processing capabilities."""
        return {cls.VISION_IMAGE_PROCESSING.value}

    @classmethod
    def requires_image_generation(cls) -> Set[str]:
        """Get the set of routes that require image generation capabilities."""
        return {cls.IMAGE_GENERATION.value}
