import logging
import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse

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
        cleaned_urls = [url.rstrip(".,;:!?") for url in urls]

        return cleaned_urls
    except Exception as e:
        logger.error(f"Error extracting URLs from text: {e}")
        return []


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate if a URL is well-formed and has a valid scheme.

    Args:
        url: The URL to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL is empty"

    try:
        # Add scheme if missing for urlparse to work
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        result = urlparse(url)

        # Check if URL has valid scheme and netloc
        if not result.scheme or result.scheme not in ('http', 'https'):
            return False, f"Invalid scheme: {result.scheme}"

        if not result.netloc:
            return False, "Missing network location"

        # Check for valid TLD (basic check)
        if '.' not in result.netloc and result.netloc != 'localhost':
            return False, "Missing top-level domain"

        return True, None
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def extract_urls_with_validation(text: str, validate: bool = True) -> List[Tuple[str, bool]]:
    """
    Extract URLs from text with optional validation.

    This is an optimized version that returns both the URL and its validity status.

    Args:
        text: The input text to search for URLs
        validate: Whether to validate each extracted URL

    Returns:
        List of tuples (url, is_valid)

    Example:
        >>> extract_urls_with_validation("Check https://valid.com and http://bad")
        [('https://valid.com', True), ('http://bad', False)]
    """
    if not text:
        return []

    urls = extract_url(text)

    if not validate:
        return [(url, True) for url in urls]

    validated = []
    for url in urls:
        is_valid, _ = validate_url(url)
        validated.append((url, is_valid))

    return validated


def get_domain_from_url(url: str) -> Optional[str]:
    """
    Extract the domain from a URL.

    Args:
        url: The URL to extract domain from

    Returns:
        The domain string or None if invalid
    """
    if not url:
        return None

    try:
        # Add scheme if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        result = urlparse(url)
        return result.netloc or None
    except Exception:
        return None


def is_same_domain(url1: str, url2: str) -> bool:
    """
    Check if two URLs belong to the same domain.

    Args:
        url1: First URL
        url2: Second URL

    Returns:
        True if both URLs have the same domain
    """
    domain1 = get_domain_from_url(url1)
    domain2 = get_domain_from_url(url2)

    if not domain1 or not domain2:
        return False

    # Extract main domain (remove www. prefix for comparison)
    def normalize_domain(domain: str) -> str:
        if domain.startswith('www.'):
            return domain[4:]
        return domain

    return normalize_domain(domain1) == normalize_domain(domain2)
