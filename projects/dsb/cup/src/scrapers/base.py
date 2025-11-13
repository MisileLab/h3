"""Base scraper interface."""

from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class PDFLink:
  """Represents a PDF link found on a webpage."""
  url: str
  title: Optional[str] = None
  post_id: Optional[str] = None
  date: Optional[str] = None


class BaseScraper(ABC):
  """Abstract base class for web scrapers."""

  @abstractmethod
  def scrape(self, url: str, limit: int = 0) -> list[PDFLink]:
    """
    Scrape PDFs from the given URL.

    Args:
        url: URL to scrape
        limit: Maximum number of PDFs to scrape (0 = no limit)

    Returns:
        List of PDFLink objects
    """
    pass

  @abstractmethod
  def download_pdf(self, pdf_link: PDFLink, output_dir: str) -> str:
    """
    Download a PDF file.

    Args:
        pdf_link: PDFLink object
        output_dir: Directory to save the PDF

    Returns:
        Path to downloaded PDF file
    """
    pass
