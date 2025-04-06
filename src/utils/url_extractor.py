import logging
import re
from typing import List, Optional

logger = logging.getLogger("vt.ai.url_extractor")

def extract_url(text: str) -> List[str]:
    """
    Extract URLs from a text string.

    This function identifies and extracts valid URLs from text input,
    including both http/https and www formats.

    Args:
        text: The input text to search for URLs

    Returns:
        A list of extracted URLs (empty list if none found or if an error occurs)

    Example:
        >>> extract_url("Check out this link: https://example.com and www.test.org")
        ['https://example.com', 'www.test.org']
    """
    if not text:
        return []

    try:
        # Regular expression pattern to match URLs
        # This matches HTTP/HTTPS URLs and www domains
        url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+(?=[\s,.;]|$)'

        # Find all URLs in the text
        urls = re.findall(url_pattern, text)

        # Remove any trailing punctuation that might have been included
        cleaned_urls = [url.rstrip('.,;:!?') for url in urls]

        return cleaned_urls
    except Exception as e:
        logger.error(f"Error extracting URLs from text: {e}")
        return []
