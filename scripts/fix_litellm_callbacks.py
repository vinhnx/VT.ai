#!/usr/bin/env python
"""
Fix script for LiteLLM callback compatibility.

This script updates the LiteLLM callback implementation in VT.ai
to be compatible with the installed LiteLLM version.
"""

import importlib
import inspect
import logging
import os
import sys
from typing import Dict, List, Optional

import dotenv
import litellm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vt.ai.fix")

# Load environment variables
dotenv.load_dotenv()


def examine_litellm_internals():
    """
    Examine LiteLLM's internals to understand the callback system.
    """
    # Get LiteLLM version
    logger.info(f"LiteLLM version: {litellm.__version__}")

    # Check if expected callback attributes exist
    has_callbacks = hasattr(litellm, "callbacks")
    has_success_callback = hasattr(litellm, "success_callback")
    has_failure_callback = hasattr(litellm, "failure_callback")

    logger.info(f"Has callbacks attribute: {has_callbacks}")
    logger.info(f"Has success_callback attribute: {has_success_callback}")
    logger.info(f"Has failure_callback attribute: {has_failure_callback}")

    # Analyze LiteLLM source code
    source_file = inspect.getfile(litellm)
    logger.info(f"LiteLLM source file: {source_file}")

    # Look for callback handler methods
    if hasattr(litellm.utils, "handle_success"):
        logger.info("Found litellm.utils.handle_success method")
    else:
        logger.info("litellm.utils.handle_success method not found")

    # Check for internal paths to examine
    litellm_dir = os.path.dirname(source_file)
    logger.info(f"LiteLLM directory: {litellm_dir}")

    for f in os.listdir(litellm_dir):
        if f.endswith(".py"):
            logger.info(f"Found module: {f}")

    # Look for success callback handler
    success_handlers = []
    for name, obj in inspect.getmembers(litellm.utils):
        if name in ["call_success_callback", "handle_success", "success_handler"]:
            success_handlers.append(name)
            logger.info(f"Found potential success handler: {name}")
            if inspect.isfunction(obj):
                logger.info(f"Signature: {inspect.signature(obj)}")

    if not success_handlers:
        logger.warning("No success callback handlers found in litellm.utils")


def fix_litellm_callback_class():
    """
    Fix the VTAISupabaseHandler class to be compatible with the current LiteLLM version.
    """
    # Define the path to the file
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "vtai", "utils", "litellm_callbacks.py"
    )

    # Check if the file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    # Read the current implementation
    with open(file_path, "r") as f:
        content = f.read()

    # Determine callback method compatibility based on LiteLLM version
    use_call_name = None

    # For newer versions of LiteLLM
    if hasattr(litellm.utils, "handle_success"):
        use_call_name = "success_handler"
    # Fallback to consistent method names
    else:
        use_call_name = "__call__"

    logger.info(f"Using callback method name: {use_call_name}")

    if use_call_name == "success_handler":
        # No need to change the implementation, it's already correct
        logger.info("Current implementation is compatible with this LiteLLM version")
        return True

    if use_call_name == "__call__":
        # Need to add a __call__ method that forwards to the appropriate method
        logger.info("Adding __call__ method to make the handler compatible with LiteLLM")

        # Check if __call__ method already exists
        if "__call__" in content:
            logger.info("__call__ method already exists")
            return True

        # Insert the __call__ method after the class definition
        class_def_end = content.find("class VTAISupabaseHandler:") + len("class VTAISupabaseHandler:")
        insertion_point = content.find("    def", class_def_end)

        call_method = """
    def __call__(self, kwargs, response_obj=None, start_time=None, end_time=None, error=None):
        """
        General callback method that dispatches to the appropriate handler method.
        This is for compatibility with older LiteLLM versions.
        """
        if error is not None:
            return self.log_failure_event(kwargs, error, start_time, end_time)
        else:
            return self.log_success_event(kwargs, response_obj, start_time, end_time)
"""

        # Insert the new method
        new_content = content[:insertion_point] + call_method + content[insertion_point:]

        # Write the updated content back to the file
        with open(file_path, "w") as f:
            f.write(new_content)

        logger.info(f"Added __call__ method to {file_path}")
        return True

    return False


def fix_callback_registration():
    """
    Fix the callback registration in the initialization function.
    """
    # Define the path to the file
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "vtai", "utils", "litellm_callbacks.py"
    )

    # Check if the file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    # Read the current implementation
    with open(file_path, "r") as f:
        content = f.read()

    # Find the initialize_litellm_callbacks function
    init_function_start = content.find("def initialize_litellm_callbacks(")
    if init_function_start == -1:
        logger.error("initialize_litellm_callbacks function not found")
        return False

    init_function_end = content.find("async def", init_function_start)
    if init_function_end == -1:
        init_function_end = len(content)

    init_function = content[init_function_start:init_function_end]

    # Check current callback registration
    register_callbacks = False
    if "litellm.callbacks = [callback_handler]" in init_function:
        register_callbacks = True

    # Update the function if needed
    new_init_function = init_function

    if not register_callbacks:
        # Fix the callback registration logic based on the LiteLLM version
        lines = init_function.split("\n")
        updated_lines = []

        for line in lines:
            updated_lines.append(line)
            if "callback_handler = VTAISupabaseHandler(" in line:
                # Make sure we add all necessary registrations
                updated_lines.append("")
                updated_lines.append("        # Register the callback handler with LiteLLM")
                updated_lines.append("        litellm.callbacks = [callback_handler]")
                updated_lines.append("")
                updated_lines.append("        # Ensure success and failure callbacks are set")
                updated_lines.append("        if hasattr(litellm, \"success_callback\"):")
                updated_lines.append("            litellm.success_callback = [callback_handler]")
                updated_lines.append("        if hasattr(litellm, \"failure_callback\"):")
                updated_lines.append("            litellm.failure_callback = [callback_handler]")
                updated_lines.append("")

        new_init_function = "\n".join(updated_lines)

        # Update the content
        new_content = content[:init_function_start] + new_init_function + content[init_function_end:]

        # Write the updated content back to the file
        with open(file_path, "w") as f:
            f.write(new_content)

        logger.info(f"Updated callback registration in {file_path}")
        return True

    return True


def main():
    """Main function."""
    logger.info("Examining LiteLLM installation...")
    examine_litellm_internals()

    logger.info("\nFixing VTAISupabaseHandler class...")
    if fix_litellm_callback_class():
        logger.info("Successfully fixed VTAISupabaseHandler class")
    else:
        logger.error("Failed to fix VTAISupabaseHandler class")

    logger.info("\nFixing callback registration...")
    if fix_callback_registration():
        logger.info("Successfully fixed callback registration")
    else:
        logger.error("Failed to fix callback registration")

    logger.info("\nFix script completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
