"""
File handling utilities for VT application.

Processes file uploads and conversions for various media types.
"""

import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import chainlit as cl
import requests
from openai import AsyncOpenAI
from PIL import Image

from vtai.utils.config import allowed_mime, logger


async def check_files(files: List) -> Tuple[bool, Optional[str]]:
    """
    Check if all files are of allowed mime types

    Args:
        files: List of files to check

    Returns:
        Tuple of (is_valid, error_message)
    """
    invalid_files = []
    for file in files:
        if file.mime not in allowed_mime:
            invalid_files.append(f"{file.name} ({file.mime})")

    if invalid_files:
        error_msg = (
            f"The following files are not supported: {', '.join(invalid_files)}. "
            f"Please upload only: {', '.join(allowed_mime)}"
        )
        return False, error_msg

    return True, None


async def upload_files(files: List, async_client: AsyncOpenAI) -> List[str]:
    """
    Upload files to the OpenAI assistant API

    Args:
        files: List of files to upload
        async_client: AsyncOpenAI client instance

    Returns:
        List of file IDs from the OpenAI API
    """
    file_ids = []
    for file in files:
        uploaded_file = await async_client.files.create(
            file=Path(file.path), purpose="assistants"
        )
        file_ids.append(uploaded_file.id)
    return file_ids


async def process_files(files: List, async_client: AsyncOpenAI) -> List[str]:
    """
    Process files and upload them to the assistant if valid

    Args:
        files: List of files to process
        async_client: AsyncOpenAI client instance

    Returns:
        List of file IDs
    """
    file_ids = []
    if not files:
        return file_ids

    is_valid, error_message = await check_files(files)
    if not is_valid:
        await cl.Message(content=error_message).send()
        logger.warning(f"Invalid files rejected: {error_message}")
        return file_ids

    try:
        file_ids = await upload_files(files, async_client)
        logger.info(f"Successfully uploaded {len(file_ids)} files")
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        # Note: We'll use the error handler from error_handlers module
        # This will be added in the next steps
        return file_ids

    return file_ids


def get_file_info(file_path: str) -> Dict[str, Union[str, int]]:
    """
    Get information about a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file information (name, size, path)
    """
    try:
        file_stats = os.stat(file_path)
        return {
            "name": os.path.basename(file_path),
            "size": file_stats.st_size,
            "path": file_path,
        }
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return {"name": os.path.basename(file_path), "size": 0, "path": file_path}


def process_image_url(image_url: str) -> str:
    """
    Process an image URL to ensure it's properly formatted and accessible.

    Args:
        image_url: URL of the image

    Returns:
        Processed image URL
    """
    try:
        # If it's a data URL, return as is
        if image_url.startswith("data:image/"):
            return image_url

        # If it's a regular URL, check if it's accessible
        response = requests.head(image_url, timeout=5)
        if response.status_code == 200:
            return image_url
        else:
            logger.warning(f"Image URL returned status code {response.status_code}")
            return image_url
    except Exception as e:
        logger.error(f"Error processing image URL: {e}")
        return image_url


def validate_image_dimensions(
    image_path: str,
) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """
    Validate that an image has reasonable dimensions for processing.

    Args:
        image_path: Path to the image

    Returns:
        Tuple of (is_valid, error_message, dimensions)
    """
    try:
        max_width = 4096
        max_height = 4096

        # Check if it's a local file or URL
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path, stream=True, timeout=5)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_path)

        width, height = img.size

        # Check if dimensions are too large
        if width > max_width or height > max_height:
            error_msg = f"Image dimensions ({width}x{height}) exceed maximum allowed size ({max_width}x{max_height})"
            return False, error_msg, (width, height)

        return True, None, (width, height)
    except Exception as e:
        logger.error(f"Error validating image dimensions: {e}")
        return False, f"Error validating image: {str(e)}", None
