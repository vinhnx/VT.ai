from typing import List

import chainlit as cl


def build_llm_profile(
    icon_path: str = "./vtai/resources/vt.jpg",
) -> List[cl.ChatProfile]:
    """
    Builds LLM profiles with a consistent icon for both display names and model IDs.

    This function creates chat profiles for both friendly display names and actual model IDs,
    ensuring that icons are displayed correctly regardless of how the model is referenced.

    Args:
        icon_path: Path to the icon file to use for all profiles

    Returns:
        List of chat profiles for all models
    """
    # Import model names and IDs here to avoid circular import
    from vtai.utils.llm_providers_config import MODEL_ALIAS_MAP, MODELS, NAMES

    profiles = []

    # Create profiles for friendly display names
    for model_name in NAMES:
        profiles.append(
            cl.ChatProfile(
                name=model_name,
                icon=icon_path,
                markdown_description=f"The underlying LLM model is **{model_name}**.",
            )
        )

    # Create profiles for actual model IDs to ensure icons show when models are referenced by ID
    for model_id in MODELS:
        # Find the friendly name for this model ID if possible
        friendly_name = next(
            (name for name, id in MODEL_ALIAS_MAP.items() if id == model_id), model_id
        )

        profiles.append(
            cl.ChatProfile(
                name=model_id,
                icon=icon_path,
                markdown_description=f"Model: **{friendly_name}**",
            )
        )

    return profiles
