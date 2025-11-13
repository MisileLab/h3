"""Database module for storing PDF processing results."""

from .schema import init_database
from .repository import PDFRepository

__all__ = ["init_database", "PDFRepository"]
