from typing import Dict

import chainlit as cl


def build_llm_profile(icons_provider_map: Dict[str, str]):
    for model_name, icon_path in icons_provider_map.items():
        return cl.ChatProfile(
            name=model_name,
            icon=icon_path,
            markdown_description=f"The underlying LLM model is **{model_name}**.",
        )
