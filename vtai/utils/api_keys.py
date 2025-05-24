"""
API key management for BYOK (Bring Your Own Key) support.

- User API keys are NEVER stored in Supabase or any backend database.
- All BYOK keys are encrypted and stored ONLY in the local Chainlit session for the current session.
- For persistent local storage (CLI/desktop), use the Python keyring package: https://pypi.org/project/keyring/
  This uses the OS keychain for secure storage and is the recommended best practice for non-web use cases.
- For web/session-based apps, session encryption (as implemented here) is standard and secure.
- Keys are only decrypted in memory when needed for LLM calls, and never logged or sent to analytics.

Handles encryption and retrieval of user LLM API keys locally (not stored in Supabase).
"""

import os

from cryptography.fernet import Fernet

from vtai.utils.config import logger


# Encryption key for Fernet (should be set in env and kept secret)
def get_encryption_key() -> bytes:
    key = os.environ.get("ENCRYPTION_KEY")
    if not key:
        key = Fernet.generate_key()
        logger.warning(
            "ENCRYPTION_KEY not set. Generated a new key for this session. "
            "This is NOT safe for production. Please set ENCRYPTION_KEY in your environment for secure, persistent encryption."
        )
        os.environ["ENCRYPTION_KEY"] = key.decode()
    else:
        key = key.encode()
    return key


cipher_suite = Fernet(get_encryption_key())


def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for local session storage."""
    return cipher_suite.encrypt(api_key.encode()).decode()


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key from local session storage."""
    return cipher_suite.decrypt(encrypted_key.encode()).decode()
    return cipher_suite.decrypt(encrypted_key.encode()).decode()
