"""Main application class for PDF processing."""

import csv
from io import StringIO
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, final

from rich.console import Console
from rich.panel import Panel

from .core.config import Config
from .core.exceptions import PDFProcessingError, ConfigurationError
from .core.types import ProcessingResult, RestaurantRecord
from .extractors import DirectTextExtractor, OCRTextExtractor
from .output import CSVFormatter, TextFormatter, JSONFormatter
from .processors import LLMProcessor
from .db import init_database, PDFRepository
from .services.restaurant_locator import RestaurantLocator

console = Console()


@final
class PDFProcessor:
  """Main PDF processing application."""
  
  def __init__(
    self,
    use_ocr: bool = False,
    openai_api_key: Optional[str] = None,
    ref_wtm_x: Optional[float] = None,
    ref_wtm_y: Optional[float] = None,
    db_path: Optional[str] = None,
    store_in_db: bool = False
  ) -> None:
    """
    Initialize the PDF processor.

    Args:
        use_ocr: Whether to use OCR for text extraction
        openai_api_key: OpenAI API key for AI post-processing
        ref_wtm_x: Reference X coordinate in WTM format for address lookup
        ref_wtm_y: Reference Y coordinate in WTM format for address lookup
        db_path: Path to SQLite database file
        store_in_db: Whether to store results in database
    """
    self.use_ocr: bool = use_ocr
    self.openai_api_key: Optional[str] = openai_api_key or Config.get_openai_api_key()
    self.ref_wtm_x: Optional[float] = ref_wtm_x
    self.ref_wtm_y: Optional[float] = ref_wtm_y
    self.db_path: Optional[str] = db_path
    self.store_in_db: bool = store_in_db
    
    # Initialize extractor
    if use_ocr:
      self.extractor: DirectTextExtractor | OCRTextExtractor = OCRTextExtractor()
    else:
      self.extractor: DirectTextExtractor | OCRTextExtractor = DirectTextExtractor()
    
    # Initialize output formatters with reference coordinates
    self.csv_formatter: CSVFormatter = CSVFormatter(
      ref_wtm_x=ref_wtm_x,
      ref_wtm_y=ref_wtm_y,
      enable_address_lookup=False,
    )
    self.text_formatter: TextFormatter = TextFormatter(ref_wtm_x=ref_wtm_x, ref_wtm_y=ref_wtm_y)
    self.json_formatter: JSONFormatter = JSONFormatter(ref_wtm_x=ref_wtm_x, ref_wtm_y=ref_wtm_y)
    
    # Initialize AI processor if API key is provided
    self.ai_processor: Optional[LLMProcessor] = None
    if self.openai_api_key and Config.validate_openai_key(self.openai_api_key):
      self.ai_processor = LLMProcessor(self.openai_api_key, names_only=True)

    # Initialize database if needed
    self.repository: Optional[PDFRepository] = None
    if self.store_in_db and self.db_path:
      init_database(self.db_path)
      self.repository = PDFRepository(self.db_path)
  
  def process_pdf(self, pdf_path: str, output_config: dict) -> None:
    """Process a PDF file with the given configuration."""
    # Validate inputs
    if not os.path.exists(pdf_path):
      raise PDFProcessingError(f"PDF file not found: {pdf_path}")
    
    # Check for KAKAO_API_KEY
    kakao_api_key = os.getenv("KAKAO_API_KEY")
    if not kakao_api_key:
      console.print("⚠️ KAKAO_API_KEY environment variable not set. Address lookup will not work.", style="yellow")
    else:
      console.print(f"✅ KAKAO_API_KEY found: {kakao_api_key[:5]}...{kakao_api_key[-5:]}", style="green")
      if self.ref_wtm_x is not None and self.ref_wtm_y is not None:
        console.print(f"✅ Reference coordinates set: ({self.ref_wtm_x}, {self.ref_wtm_y})", style="green")
    
    # Set output directory
    output_dir = Config.get_output_dir(pdf_path, output_config.get("output_dir"))
    os.makedirs(output_dir, exist_ok=True)
    
    try:
      # Extract text and tables
      text_pages = self.extractor.extract_text(pdf_path)
      table_pages = []
      
      if output_config.get("include_tables", True):
        table_pages = self.extractor.extract_tables(pdf_path)
      
      # Combine text and tables by page
      combined_pages = self._combine_pages(text_pages, table_pages)
      
      # Create processing result
      result = self._create_processing_result(pdf_path, combined_pages)
      
      # Show preview if requested
      if output_config.get("show_preview", False):
        self._show_preview(result)

      # Run LLM post-processing BEFORE saving outputs
      llm_csv_content = None
      source_url = output_config.get("source_url")
      if self.ai_processor:
        llm_csv_content = self._run_llm_post_processing(result, source_url)

      # Save in requested format(s)
      output_format = output_config.get("format", "csv")
      self._save_outputs(result, output_dir, output_format, llm_csv_content=llm_csv_content)

      # Store in database if enabled
      if self.store_in_db and self.repository:
        try:
          source_url = output_config.get("source_url")
          download_date = output_config.get("download_date")
          document_id = self.repository.save_processing_result(
            result, source_url=source_url, download_date=download_date
          )
          console.print(f"✅ Results stored in database (document ID: {document_id})", style="green")
        except (RuntimeError, ValueError) as e:
          console.print(f"⚠️ Failed to store results in database: {e}", style="yellow")

      # Show summary
      self._show_summary(result, output_dir)
      
    except (OSError, IOError, ValueError) as e:
      raise PDFProcessingError(f"Error during processing: {e}") from e
  
  def _combine_pages(self, text_pages: list, table_pages: list) -> list:
    """Combine text and table pages."""
    # Create a map of pages by page number
    page_map = {}
    
    # Add text pages
    for page in text_pages:
      page_map[page.page] = page
    
    # Add table pages (merge with existing pages if they exist)
    for page in table_pages:
      if page.page in page_map:
        # Merge tables into existing page
        page_map[page.page].tables.extend(page.tables)
      else:
        # Create new page with just tables
        page_map[page.page] = page
    
    # Return sorted pages
    return sorted(page_map.values(), key=lambda p: p.page)
  
  def _create_processing_result(self, pdf_path: str, pages: list):
    """Create a ProcessingResult from pages."""
    total_pages = len(pages)
    total_text_lines = sum(len(page.text_lines) for page in pages)
    total_tables = sum(len(page.tables) for page in pages)
    return ProcessingResult(
      pdf_path=pdf_path,
      pages=pages,
      total_pages=total_pages,
      total_text_lines=total_text_lines,
      total_tables=total_tables,
      extraction_method=self.extractor.__class__.__name__,
      processing_time=0.0  # Will be set by the extractor
    )
  
  def _show_preview(self, result) -> None:
    """Show preview of extracted data."""
    console.print("\n📋 Preview of extracted data:", style="blue")
    
    # Show text preview
    if result.pages:
      console.print("\n📝 Text Preview:", style="yellow")
      for page in result.pages[:2]:  # Show first 2 pages
        console.print(f"Page {page.page} ({page.total_lines} lines):")
        for line in page.text_lines[:5]:  # Show first 5 lines
          text = line.text[:100]  # Truncate long text
          console.print(f"  {line.line_number}: {text}")
        console.print("")
    
    # Show table preview
    all_tables = []
    for page in result.pages:
      all_tables.extend(page.tables)
    
    if all_tables:
      console.print("📊 Table Preview:", style="yellow")
      for table in all_tables[:2]:  # Show first 2 tables
        console.print(f"Table {table.table_index} ({table.shape[0]} rows, {table.shape[1]} columns):")
        if table.rows:
          # Show first few rows
          for row_idx, row in enumerate(table.rows[:3], 1):
            row_text = " | ".join(f"{k}: {v}" for k, v in list(row.items())[:3])
            console.print(f"  Row {row_idx}: {row_text}")
        console.print("")
  
  def _save_outputs(self, result, output_dir: str, output_format: str, llm_csv_content: str | None = None) -> None:
    """Save outputs in the requested format(s)."""
    pdf_name = Path(result.pdf_path).stem

    if output_format.lower() in ["csv", "all"]:
      csv_path = Config.get_output_path(result.pdf_path, output_dir, "csv")

      # Use LLM-processed CSV if available, otherwise use standard extraction
      if llm_csv_content:
        with open(csv_path, "w", encoding="utf-8") as f:
          f.write(llm_csv_content)
        console.print(f"✅ LLM-processed CSV saved to: {csv_path}", style="green")
      else:
        self.csv_formatter.save(result, csv_path)

    if output_format.lower() in ["txt", "all"]:
      txt_path = Config.get_output_path(result.pdf_path, output_dir, "txt")
      self.text_formatter.save(result, txt_path)

    if output_format.lower() in ["json", "all"]:
      json_path = Config.get_output_path(result.pdf_path, output_dir, "json")
      self.json_formatter.save(result, json_path)
  
  def _show_summary(self, result, output_dir: str) -> None:
    """Show processing summary."""
    console.print("\n📊 Processing Summary:", style="blue")
    console.print(f"  • Pages processed: {result.total_pages}")
    console.print(f"  • Text lines extracted: {result.total_text_lines}")
    console.print(f"  • Tables detected: {result.total_tables}")
    console.print(f"  • Output directory: {output_dir}")
    console.print(f"  • Extraction method: {result.extraction_method}")

  def _run_llm_post_processing(
    self,
    result: ProcessingResult,
    source_url: str | None = None,
  ) -> str | None:
    """
    Run LLM post-processing and return CSV content.

    Args:
        result: The processing result to process with LLM
        source_url: Optional source URL for the PDF

    Returns:
        CSV content as string, or None if processing fails or is not available
    """
    if not self.ai_processor:
      console.print(
        "⚠️  OPENAI_API_KEY not set or invalid. Skipping LLM post-processing.",
        style="yellow",
      )
      return None

    try:
      console.print("🤖 Running LLM post-processing...", style="blue")
      csv_content = self.ai_processor.process(result)

      # Extract restaurant names and store them if we have a repository
      restaurant_names = self._extract_restaurant_names_from_csv(csv_content)
      if restaurant_names and self.repository:
        self._search_and_store_restaurants(restaurant_names, result.pdf_path, source_url)
      elif restaurant_names and not self.repository:
        console.print(f"📝 Found {len(restaurant_names)} restaurant names (database storage not enabled)", style="blue")

      return csv_content
    except (PDFProcessingError, ValueError) as e:
      console.print(f"⚠️  LLM post-processing error: {e}", style="yellow")
      return None

  def _maybe_run_llm_post_processing(
    self,
    result: ProcessingResult,
    output_dir: str,
    custom_output_path: str | None = None,
    source_url: str | None = None,
  ) -> None:
    """Invoke LLM post-processing when OCR is enabled."""
    if not self.use_ocr:
      return

    if not self.ai_processor:
      console.print(
        "⚠️  OPENAI_API_KEY not set or invalid. Skipping LLM post-processing despite OCR mode.",
        style="yellow",
      )
      return

    llm_output_path = custom_output_path or self._default_llm_output_path(result.pdf_path, output_dir)

    try:
      csv_content = self.ai_processor.process(result)
      with open(llm_output_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
      console.print(f"🤖 LLM-processed CSV saved to: {llm_output_path}", style="green")

      restaurant_names = self._extract_restaurant_names_from_csv(csv_content)
      if restaurant_names:
        self._search_and_store_restaurants(restaurant_names, result.pdf_path, source_url)
      else:
        console.print("⚠️  No restaurant names detected in LLM output.", style="yellow")
    except (OSError, IOError, ValueError) as e:
      console.print(f"⚠️  Failed to save LLM-processed output: {e}", style="yellow")
    except PDFProcessingError as e:  # pragma: no cover - defensive
      console.print(f"⚠️  LLM post-processing error: {e}", style="yellow")

  def _default_llm_output_path(self, pdf_path: str, output_dir: str) -> str:
    pdf_name = Path(pdf_path).stem
    return str(Path(output_dir) / f"{pdf_name}_llm.csv")

  def _extract_restaurant_names_from_csv(self, csv_content: str) -> list[str]:
    csv_content = csv_content.strip()
    if not csv_content:
      return []

    reader = csv.reader(StringIO(csv_content))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
      return []

    keywords = ["restaurant", "store", "가맹점", "상호", "매장", "place"]
    header_row = rows[0]
    has_header = any(
      any(keyword in cell.strip().lower() for keyword in keywords)
      for cell in header_row
    )

    data_rows = rows[1:] if has_header else rows
    target_idx = 0
    if has_header:
      for idx, cell in enumerate(header_row):
        col_lower = cell.strip().lower()
        if any(keyword in col_lower for keyword in keywords):
          target_idx = idx
          break

    names: list[str] = []
    for row in data_rows:
      if target_idx >= len(row):
        continue
      name = row[target_idx].strip().strip('"')
      if name and name not in names:
        names.append(name)

    return names

  def _search_and_store_restaurants(
    self,
    names: list[str],
    pdf_path: str,
    source_url: str | None,
  ) -> None:
    if not names:
      return

    if not self.repository:
      console.print("⚠️  Repository not configured; skipping restaurant storage.", style="yellow")
      return

    locator = RestaurantLocator(
      api_key=os.getenv("KAKAO_API_KEY"),
      openai_api_key=self.openai_api_key or os.getenv("OPENAI_API_KEY"),
      ref_wtm_x=self.ref_wtm_x,
      ref_wtm_y=self.ref_wtm_y,
    )
    records = locator.lookup(names, source_pdf=pdf_path, source_url=source_url)

    saved = self.repository.save_restaurants(records)
    console.print(f"📥 Saved {saved} restaurant records to database", style="green")
  
  def process_with_ai(self, pdf_path: str, output_path: str) -> None:
    """Process PDF with AI post-processing."""
    if not self.ai_processor:
      raise ConfigurationError("OpenAI API key not configured")
    
    # Extract data
    text_pages = self.extractor.extract_text(pdf_path)
    table_pages = self.extractor.extract_tables(pdf_path)
    combined_pages = self._combine_pages(text_pages, table_pages)
    result = self._create_processing_result(pdf_path, combined_pages)
    
    # Process with AI
    csv_content = self.ai_processor.process(result)
    
    # Save result
    try:
      with open(output_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
      console.print(f"✅ AI-processed CSV saved to: {output_path}", style="green")
    except (OSError, IOError, ValueError) as e:
      raise PDFProcessingError(f"Error saving AI-processed CSV: {e}") from e 
