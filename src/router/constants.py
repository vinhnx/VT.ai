from enum import Enum


class SemanticRouterType(str, Enum):
    """
    We use https://github.com/aurelio-labs/semantic-router?tab=readme-ov-file

    Semantic Router is a superfast decision-making layer for your LLMs and agents. Rather than waiting for slow LLM generations to make tool-use decisions, we use the magic of semantic vector space to make those decisions — routing our requests using semantic meaning.

    --

    The definition is in `layers.json` file. Here we define routes' name constant.

    It' must be mapped together.
    """

    IMAGE_GENERATION = "image-generation"
    TEXT_PROCESSING = "text-processing"
    CASUAL_CONVERSATION = "casual-conversation"
    CURIOUS = "curious"
    VISION_IMAGE_PROCESSING = "vision-image-processing"
