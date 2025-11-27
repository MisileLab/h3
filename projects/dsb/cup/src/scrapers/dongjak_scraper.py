"""
Scraper for Dongjak government website.
URL: https://www.dongjak.go.kr/portal/bbs/B0000591/list.do?menuNo=200209
"""

import os
import re
import time
from pathlib import Path
from typing import Optional, final
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseScraper, PDFLink


console = Console()


@final
class DongjakScraper(BaseScraper):
  """Scraper for Dongjak government portal."""

  def __init__(
    self,
    use_selenium: bool = False,
    max_pages: Optional[int] = None,
    request_delay: float = 0.5,
  ) -> None:
    """
    Initialize scraper.

    Args:
        use_selenium: Whether to use Selenium (fallback if requests fails)
        max_pages: Maximum number of pages to scrape (None for unlimited)
        request_delay: Delay between requests in seconds
    """
    self.use_selenium = use_selenium
    self.max_pages = max_pages
    self.request_delay = request_delay
    self.session = requests.Session()
    self.session.headers.update({
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

  def scrape(self, url: str, limit: int = 0) -> list[PDFLink]:
    """
    Scrape PDF links from Dongjak portal.

    Args:
        url: Base URL of the bulletin board
        limit: Maximum number of PDFs to scrape

    Returns:
        List of PDFLink objects
    """
    pdf_links: list[PDFLink] = []
    page = 1

    console.print(f"[yellow]Starting to scrape: {url}[/yellow]")

    with Progress(
      SpinnerColumn(),
      TextColumn("[progress.description]{task.description}"),
      console=console,
    ) as progress:
      task = progress.add_task("Scraping pages...", total=None)

      while True:
        progress.update(task, description=f"Scraping page {page}...")

        # Build page URL
        page_url = f"{url}&pageIndex={page}" if "?" in url else f"{url}?pageIndex={page}"

        try:
          response = self._fetch_page(page_url)
        except requests.RequestException as e:
          console.print(f"[red]Failed to fetch page {page} after retries: {e}[/red]")
          if self.use_selenium:
            console.print("[yellow]Falling back to Selenium...[/yellow]")
            # TODO: Implement Selenium fallback if needed
          break

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all post rows
        posts = soup.select("tbody tr") or soup.select("tr.list-row")

        if not posts:
          console.print(f"[yellow]No posts found on page {page}, stopping.[/yellow]")
          break

        # Extract PDF links from each post
        page_pdf_count = 0
        for post in posts:
          # Find title link
          title_link = post.select_one("td.title a") or post.select_one("a.title")
          if not title_link:
            continue

          title = title_link.get_text(strip=True)
          post_url = title_link.get("href", "")

          if not post_url:
            continue

          # Make absolute URL
          if not post_url.startswith("http"):
            base_url = "/".join(url.split("/")[:3])  # Get domain
            post_url = urljoin(base_url, post_url)

          # Extract post ID
          post_id = self._extract_post_id(post_url)

          # Find PDF attachments in post
          post_pdfs = self._extract_pdfs_from_post(post_url, title, post_id)
          pdf_links.extend(post_pdfs)
          page_pdf_count += len(post_pdfs)

          # Check limit
          if limit > 0 and len(pdf_links) >= limit:
            console.print(f"[green]Reached limit of {limit} PDFs[/green]")
            return pdf_links[:limit]

        console.print(f"[blue]Page {page}: Found {page_pdf_count} PDFs[/blue]")

        # Move to next page
        page += 1

        # Check page limit
        if self.max_pages is not None and page > self.max_pages:
          console.print(f"[yellow]Reached page limit ({self.max_pages}), stopping.[/yellow]")
          break

        # Small delay to be polite
        time.sleep(self.request_delay)

    console.print(f"[green]Total PDFs found: {len(pdf_links)}[/green]")
    return pdf_links

  @retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
  )
  def _fetch_page(self, url: str, timeout: int = 10) -> requests.Response:
    """
    Fetch a page with retry logic.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Response object

    Raises:
        requests.RequestException: If all retry attempts fail
    """
    response = self.session.get(url, timeout=timeout)
    response.raise_for_status()
    return response

  def _extract_post_id(self, post_url: str) -> Optional[str]:
    """Extract post ID from URL."""
    parsed = urlparse(post_url)
    query_params = parse_qs(parsed.query)
    return query_params.get("nttId", [None])[0] or query_params.get("id", [None])[0]

  def _extract_pdfs_from_post(
    self, post_url: str, title: str, post_id: Optional[str]
  ) -> list[PDFLink]:
    """Extract PDF links from a post detail page."""
    pdf_links: list[PDFLink] = []

    try:
      response = self._fetch_page(post_url)
    except requests.RequestException as e:
      console.print(f"[red]Failed to fetch post {post_url} after retries: {e}[/red]")
      return pdf_links

    soup = BeautifulSoup(response.text, "html.parser")

    # Find attachment section - newer site version exposes downloads through
    # `fileDown.do` links that do not carry a .pdf suffix.
    attachments = soup.select(
      "a[href*='.pdf'], a.attach, a.btn-download, a[href*='fileDown']"
    )

    for attach in attachments:
      href = attach.get("href", "")
      if not href:
        continue

      href_lower = href.lower()
      is_pdf = href_lower.endswith(".pdf")
      is_download = "download" in href_lower or "filedown" in href_lower
      if not (is_pdf or is_download):
        continue

      # Make absolute URL
      if not href.startswith("http"):
        base_url = "/".join(post_url.split("/")[:3])
        href = urljoin(base_url, href)

      pdf_links.append(PDFLink(url=href, title=title, post_id=post_id))

    return pdf_links

  def download_pdf(self, pdf_link: PDFLink, output_dir: str) -> str:
    """
    Download a PDF file.

    Args:
        pdf_link: PDFLink object
        output_dir: Directory to save the PDF

    Returns:
        Path to downloaded PDF file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename = self._generate_filename(pdf_link)
    file_path = output_path / filename

    # Skip if already exists
    if file_path.exists():
      console.print(f"[yellow]Skipping existing file: {filename}[/yellow]")
      return str(file_path)

    # Download with retry logic
    try:
      response = self._fetch_page(pdf_link.url, timeout=30)

      with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)

      console.print(f"[green]Downloaded: {filename}[/green]")
      return str(file_path)

    except requests.RequestException as e:
      console.print(f"[red]Failed to download {pdf_link.url} after retries: {e}[/red]")
      raise RuntimeError(f"Failed to download PDF: {e}") from e

  def _generate_filename(self, pdf_link: PDFLink) -> str:
    """Generate a safe filename for the PDF."""
    # Try to extract filename from URL
    parsed = urlparse(pdf_link.url)
    url_filename = Path(parsed.path).name

    if url_filename and url_filename.lower().endswith(".pdf"):
      # Clean up filename
      filename = re.sub(r"[^\w\s.-]", "_", url_filename)
      return filename

    # Generate from title and post ID
    if pdf_link.title:
      title_clean = re.sub(r"[^\w\s-]", "_", pdf_link.title)[:50]
      post_id_part = f"_{pdf_link.post_id}" if pdf_link.post_id else ""
      return f"{title_clean}{post_id_part}.pdf"

    # Fallback to post ID or timestamp
    if pdf_link.post_id:
      return f"post_{pdf_link.post_id}.pdf"

    return f"document_{int(time.time())}.pdf"
