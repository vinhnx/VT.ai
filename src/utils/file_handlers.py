"""
File handling utilities for VT.ai application.

Handles file validation, processing, and uploading to various services.
"""

import logging
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chainlit as cl
from openai import AsyncOpenAI

from utils.config import allowed_mime, logger

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