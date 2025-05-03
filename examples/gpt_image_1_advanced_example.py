"""
GPT-Image-1 Advanced Usage Example

This example demonstrates how to use the advanced GPT-Image-1 features programmatically
through the VT.ai API. It shows how to generate images with various customizations.
"""

import asyncio
import os
from pathlib import Path

from vtai.utils import constants as const
from vtai.utils import llm_providers_config as conf
from vtai.utils.config import get_openai_client, logger
from vtai.utils.media_processors import handle_trigger_async_image_gen


async def generate_example_images():
    """Generate a series of example images showcasing different GPT-Image-1 features."""
    # Make sure the OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable to use this example.")
        return

    # Create output directory if it doesn't exist
    output_dir = Path("./examples/generated_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example 1: Logo Design with Transparent Background
    print("\n=== Generating Logo Design with Transparent Background ===")
    os.environ["VT_SETTINGS_IMAGE_GEN_IMAGE_SIZE"] = "1024x1024"
    os.environ["VT_SETTINGS_IMAGE_GEN_IMAGE_QUALITY"] = "high"
    os.environ["VT_SETTINGS_IMAGE_GEN_BACKGROUND"] = "transparent"
    os.environ["VT_SETTINGS_IMAGE_GEN_OUTPUT_FORMAT"] = "png"
    os.environ["VT_SETTINGS_IMAGE_GEN_MODERATION"] = "auto"

    logo_prompt = """
	Generate a minimalist logo for a tech startup called "QuantumLeap".
	The logo should feature a stylized letter 'Q' that transforms into an upward arrow.
	Use a gradient of blue to purple colors. Keep the design clean with plenty of negative space.
	Make sure it works well on both light and dark backgrounds.
	"""
    await handle_trigger_async_image_gen(logo_prompt)

    # Example 2: Realistic Product Photography
    print("\n=== Generating Realistic Product Photography ===")
    os.environ["VT_SETTINGS_IMAGE_GEN_IMAGE_SIZE"] = "1536x1024"
    os.environ["VT_SETTINGS_IMAGE_GEN_IMAGE_QUALITY"] = "high"
    os.environ["VT_SETTINGS_IMAGE_GEN_BACKGROUND"] = "auto"
    os.environ["VT_SETTINGS_IMAGE_GEN_OUTPUT_FORMAT"] = "jpeg"
    os.environ["VT_SETTINGS_IMAGE_GEN_OUTPUT_COMPRESSION"] = "90"

    product_prompt = """
	Create a professional product photograph of a modern smartphone with a sleek black design.
	The phone should be displayed at a 3/4 angle on a minimalist white surface with soft shadows.
	There should be a subtle reflection on the surface.
	The phone screen should display a colorful home screen with app icons.
	Use dramatic studio lighting with a soft blue accent light from the left side.
	"""
    await handle_trigger_async_image_gen(product_prompt)

    # Example 3: Technical Diagram with Transparent Background for Web
    print("\n=== Generating Technical Diagram ===")
    os.environ["VT_SETTINGS_IMAGE_GEN_IMAGE_SIZE"] = "1024x1024"
    os.environ["VT_SETTINGS_IMAGE_GEN_IMAGE_QUALITY"] = "high"
    os.environ["VT_SETTINGS_IMAGE_GEN_BACKGROUND"] = "transparent"
    os.environ["VT_SETTINGS_IMAGE_GEN_OUTPUT_FORMAT"] = "png"

    diagram_prompt = """
	Create a clear technical diagram of a renewable energy smart grid system.
	The diagram should include solar panels, wind turbines, energy storage systems, and a control center.
	Use a clean, modern design with a color-coded legend explaining each component.
	Include directional arrows showing energy flow throughout the system.
	Add simple, concise labels for each major component.
	The style should be professional and suitable for a business presentation.
	"""
    await handle_trigger_async_image_gen(diagram_prompt)

    # Example 4: Portrait Photography
    print("\n=== Generating Portrait Photography ===")
    os.environ["VT_SETTINGS_IMAGE_GEN_IMAGE_SIZE"] = "1024x1536"
    os.environ["VT_SETTINGS_IMAGE_GEN_IMAGE_QUALITY"] = "high"
    os.environ["VT_SETTINGS_IMAGE_GEN_BACKGROUND"] = "auto"
    os.environ["VT_SETTINGS_IMAGE_GEN_OUTPUT_FORMAT"] = "jpeg"
    os.environ["VT_SETTINGS_IMAGE_GEN_OUTPUT_COMPRESSION"] = "100"

    portrait_prompt = """
	Generate a professional studio portrait of a female CEO in her 40s with short dark hair.
	She should be wearing a tailored navy blue suit jacket over a white blouse.
	The lighting should be high-key with soft shadows and a subtle rim light.
	The background should be a gradient of light gray.
	She should have a confident, approachable expression with a slight smile.
	The composition should be from chest up, with her body turned slightly and face toward the camera.
	"""
    await handle_trigger_async_image_gen(portrait_prompt)

    print(
        "\nAll examples generated successfully. Images have been saved to the 'imgs' directory."
    )


if __name__ == "__main__":
    asyncio.run(generate_example_images())
