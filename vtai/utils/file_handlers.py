"""
File handling utilities for VT application.

Processes file uploads and conversions for various media types.
"""

import asyncio
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import chainlit as cl
import requests
from openai import AsyncOpenAI
from PIL import Image

from .config import allowed_mime, logger


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
        # error_message could be None, provide a default
        message = error_message if error_message else "Invalid file(s) rejected."
        await cl.Message(content=message).send()
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


async def process_image_url_async(image_url: str, timeout: float = 5.0) -> Tuple[bool, str]:
    """
    Process an image URL asynchronously to check if it's accessible.

    This async version avoids blocking the event loop during HTTP requests.

    Args:
        image_url: URL of the image
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_accessible, processed_url)
    """
    try:
        # If it's a data URL, return as is
        if image_url.startswith("data:image/"):
            return True, image_url

        # Use asyncio.to_thread to run synchronous requests in a thread pool
        # This prevents blocking the event loop
        response = await asyncio.to_thread(
            requests.head, image_url, timeout=timeout, allow_redirects=True
        )

        if response.status_code == 200:
            return True, image_url
        else:
            logger.warning(f"Image URL returned status code {response.status_code}")
            return False, image_url
    except requests.Timeout:
        logger.warning(f"Image URL request timed out: {image_url}")
        return False, image_url
    except requests.RequestException as e:
        logger.error(f"Error processing image URL: {e}")
        return False, image_url
    except Exception as e:
        logger.error(f"Error processing image URL: {e}")
        return False, image_url


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


async def validate_image_url_async(
    image_url: str,
    max_size_mb: float = 10.0,
    timeout: float = 10.0
) -> Tuple[bool, Optional[str], Optional[Dict[str, int]]]:
    """
    Validate an image URL asynchronously - check accessibility and get dimensions.

    This is an optimized async version that performs all validation in a non-blocking way.

    Args:
        image_url: URL of the image
        max_size_mb: Maximum allowed file size in MB
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_valid, error_message, dimensions_dict)
    """
    try:
        # If it's a data URL, we can't validate size/dimensions easily
        if image_url.startswith("data:image/"):
            return True, None, None

        # Check accessibility first
        is_accessible, _ = await process_image_url_async(image_url, timeout=timeout / 2)
        if not is_accessible:
            return False, f"Image URL is not accessible", None

        # Get the image to check size and dimensions
        async def fetch_image():
            return await asyncio.to_thread(
                requests.get, image_url, stream=True, timeout=timeout
            )

        response = await fetch_image()

        # Check content-length if available
        content_length = response.headers.get('content-length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > max_size_mb:
                return False, f"Image size ({size_mb:.1f}MB) exceeds maximum ({max_size_mb}MB)", None

        # Get image dimensions
        try:
            img = Image.open(BytesIO(response.content))
            width, height = img.size

            max_dimension = 8192  # Common limit for most vision models
            if width > max_dimension or height > max_dimension:
                return False, f"Image dimensions ({width}x{height}) exceed maximum ({max_dimension}x{max_dimension})", None

            return True, None, {"width": width, "height": height}
        except Exception as e:
            # If we can't open as image, just return accessible status
            logger.warning(f"Could not validate image dimensions: {e}")
            return True, None, None

    except Exception as e:
        logger.error(f"Error validating image URL: {e}")
        return False, f"Validation error: {str(e)}", None


async def validate_image_url(image_url: str) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """
    Validate that an image has reasonable dimensions for processing.

    This is a compatibility wrapper that calls the async version.

    Args:
        image_url: Path or URL to the image

    Returns:
        Tuple of (is_valid, error_message, dimensions)
    """
    is_valid, error_msg, dims = await validate_image_url_async(image_url)

    if dims:
        return is_valid, error_msg, (dims["width"], dims["height"])
    return is_valid, error_msg, None


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
