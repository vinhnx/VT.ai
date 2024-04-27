from typing import Dict

import chainlit as cl


async def build_llm_profile(icons_provider_map: Dict[str, str]):
    for model_name, icon_path in icons_provider_map.items():
        await cl.Avatar(name=model_name, path=icon_path).send()
