from typing import Dict
import chainlit as cl


async def build_llm_profile(icons_provider_map: Dict[str, str]):
    for key, value in icons_provider_map.items():
        await cl.Avatar(name=key, path=value).send()
