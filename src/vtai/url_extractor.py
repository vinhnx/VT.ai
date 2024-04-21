import re


def extract_url(text):
    # Regular expression pattern to match URLs
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'

    # Find all URLs in the text
    urls = re.findall(url_pattern, text)

    return urls
