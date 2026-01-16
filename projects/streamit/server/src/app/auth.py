"""
Simple token authentication for viewer connections.
"""

import hashlib
from app.config import settings
from app.logging_setup import logger


def validate_token(token: str) -> bool:
    """
    Validate viewer token against configured tokens.

    Args:
        token: Bearer token from client

    Returns:
        True if token is valid, False otherwise
    """
    if not token:
        return False

    valid_tokens = settings.get_valid_tokens()
    if not valid_tokens:
        logger.warning("No viewer tokens configured")
        return False

    is_valid = token in valid_tokens
    if not is_valid:
        logger.warning(f"Invalid token attempt: {hash_token(token)}")

    return is_valid


def hash_token(token: str) -> str:
    """
    Hash token for safe logging.

    Args:
        token: Plain token

    Returns:
        SHA256 hash of token (first 8 chars)
    """
    return hashlib.sha256(token.encode()).hexdigest()[:8]
