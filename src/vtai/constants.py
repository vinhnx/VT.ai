from enum import Enum


class SemanticRouterType(str, Enum):
    """
    We use https://github.com/aurelio-labs/semantic-router?tab=readme-ov-file

    Semantic Router is a superfast decision-making layer for your LLMs and agents. Rather than waiting for slow LLM generations to make tool-use decisions, we use the magic of semantic vector space to make those decisions — routing our requests using semantic meaning.

    --

    The definition is in `semantic_route_layers.json` file. Here we define routes' name constant.

    It' must be mapped together.
    """

    IMAGE_GEN = "image-gen"
    CHITCHAT = "chitchat"
    GREETINGS = "greetings"
    SMALL_TALK = "small_talk"
