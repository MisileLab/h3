"""
Metering for audio usage (seconds-based).
"""

from collections import defaultdict
from typing import Optional
import logging
from datetime import date

logger = logging.getLogger(__name__)


class UsageMeter:
    """Track usage per token in seconds."""

    _daily_usage: dict[str, float] = defaultdict(float)
    _last_reset: date = date.today()

    @classmethod
    def reset_if_new_day(cls):
        """Reset daily usage if a new day has started."""
        today = date.today()
        if cls._last_reset < today:
            logger.info(f"Resetting daily usage. Previous: {cls._daily_usage}")
            cls._daily_usage.clear()
            cls._last_reset = today

    @classmethod
    def add_bytes(cls, token: str, bytes_count: int) -> float:
        """
        Add audio bytes and return seconds used.

        Args:
            token: User token
            bytes_count: Number of PCM16 bytes received

        Returns:
            Total seconds used for this token today
        """
        cls.reset_if_new_day()

        seconds = bytes_count / 48000.0
        cls._daily_usage[token] += seconds

        return cls._daily_usage[token]

    @classmethod
    def get_usage(cls, token: str) -> float:
        """Get current usage for token."""
        cls.reset_if_new_day()
        return cls._daily_usage[token]

    @classmethod
    def check_limit(cls, token: str, max_seconds: Optional[int]) -> bool:
        """
        Check if token has exceeded usage limit.

        Args:
            token: User token
            max_seconds: Maximum allowed seconds (None = no limit)

        Returns:
            True if within limit, False if exceeded
        """
        if max_seconds is None:
            return True

        current = cls.get_usage(token)
        if current >= max_seconds:
            logger.warning(
                f"Token {token[:8]} exceeded limit: {current:.1f}s >= {max_seconds}s"
            )
            return False

        return True

    @classmethod
    def record_session(cls, token: str, seconds: float):
        """Log session completion."""
        logger.info(
            f"Session completed - Token: {token[:8]}, "
            f"Seconds: {seconds:.2f}, Total today: {cls.get_usage(token):.2f}"
        )
