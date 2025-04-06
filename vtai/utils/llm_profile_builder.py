from typing import Dict, List

import chainlit as cl


def build_llm_profile(icons_provider_map: Dict[str, str]) -> List[cl.ChatProfile]:
    """
    Builds LLM profiles for all models in the provided map.

    Args:
        icons_provider_map: Dictionary mapping model names to their icon paths

    Returns:
        List of chat profiles for all models
    """
    profiles = []
    for model_name, icon_path in icons_provider_map.items():
        profiles.append(
            cl.ChatProfile(
                name=model_name,
                icon=icon_path,
                markdown_description=f"The underlying LLM model is **{model_name}**.",
            )
        )
    return profiles
