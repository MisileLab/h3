"""Web scrapers for downloading PDFs from government websites."""

from .base import BaseScraper
from .dongjak_scraper import DongjakScraper

__all__ = ["BaseScraper", "DongjakScraper"]
