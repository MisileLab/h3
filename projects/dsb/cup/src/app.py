"""
Main application class for PDF processing.
"""

import os
import time
from pathlib import Path
from typing import Optional, final

from rich.console import Console
from rich.panel import Panel

from .core.config import Config
from .core.exceptions import PDFProcessingError, ConfigurationError
from .core.types import ProcessingResult
from .extractors import DirectTextExtractor, OCRTextExtractor
from .output import CSVFormatter, TextFormatter, JSONFormatter
from .processors import LLMProcessor

console = Console()


@final
class PDFProcessor:
  """Main PDF processing application."""
  
  def __init__(
    self, 
    use_ocr: bool = False, 
    openai_api_key: Optional[str] = None,
    ref_wtm_x: Optional[float] = None,
    ref_wtm_y: Optional[float] = None
  ) -> None:
    """
    Initialize the PDF processor.
    
    Args:
        use_ocr: Whether to use OCR for text extraction
        openai_api_key: OpenAI API key for AI post-processing
        ref_wtm_x: Reference X coordinate in WTM format for address lookup
        ref_wtm_y: Reference Y coordinate in WTM format for address lookup
    """
    self.use_ocr: bool = use_ocr
    self.openai_api_key: Optional[str] = openai_api_key or Config.get_openai_api_key()
    self.ref_wtm_x: Optional[float] = ref_wtm_x
    self.ref_wtm_y: Optional[float] = ref_wtm_y
    
    # Initialize extractor
    if use_ocr:
      self.extractor: DirectTextExtractor | OCRTextExtractor = OCRTextExtractor()
    else:
      self.extractor: DirectTextExtractor | OCRTextExtractor = DirectTextExtractor()
    
    # Initialize output formatters with reference coordinates
    self.csv_formatter: CSVFormatter = CSVFormatter(ref_wtm_x=ref_wtm_x, ref_wtm_y=ref_wtm_y)
    self.text_formatter: TextFormatter = TextFormatter(ref_wtm_x=ref_wtm_x, ref_wtm_y=ref_wtm_y)
    self.json_formatter: JSONFormatter = JSONFormatter(ref_wtm_x=ref_wtm_x, ref_wtm_y=ref_wtm_y)
    
    # Initialize AI processor if API key is provided
    self.ai_processor: Optional[LLMProcessor] = None
    # Use OpenRouter if configured
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key and Config.validate_openrouter_key(openrouter_key):
      self.ai_processor = LLMProcessor(openrouter_key)
  
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
      
      # Save in requested format(s)
      output_format = output_config.get("format", "csv")
      self._save_outputs(result, output_dir, output_format)
      
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
  
  def _save_outputs(self, result, output_dir: str, output_format: str) -> None:
    """Save outputs in the requested format(s)."""
    pdf_name = Path(result.pdf_path).stem
    
    if output_format.lower() in ["csv", "all"]:
      csv_path = Config.get_output_path(result.pdf_path, output_dir, "csv")
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
  
  def process_with_ai(self, pdf_path: str, output_path: str) -> None:
    """Process PDF with AI post-processing."""
    if not self.ai_processor:
      raise ConfigurationError("OpenRouter API key not configured")
    
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