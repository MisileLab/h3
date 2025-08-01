"""
Base extractor class for PDF processing.
"""

from abc import ABC, abstractmethod

from ..core.types import PageData, ProcessingResult
from ..core.exceptions import PDFProcessingError


class BaseExtractor(ABC):
  """Base class for PDF extractors."""
  
  def __init__(self) -> None:
    """Initialize the extractor."""
    pass
  
  @abstractmethod
  def extract_text(self, pdf_path: str) -> list[PageData]:
    """Extract text from PDF file."""
    pass
  
  @abstractmethod
  def extract_tables(self, pdf_path: str) -> list[PageData]:
    """Extract tables from PDF file."""
    pass
  
  def process(self, pdf_path: str) -> ProcessingResult:
    # sourcery skip: extract-method
    """Process PDF and return complete result."""
    import time
    
    start_time = time.time()
    
    try:
      pages = self.extract_text(pdf_path)
      total_pages = len(pages)
      total_text_lines = sum(len(page.text_lines) for page in pages)
      total_tables = sum(len(page.tables) for page in pages)
      
      processing_time = time.time() - start_time
      
      return ProcessingResult(
        pdf_path=pdf_path,
        pages=pages,
        total_pages=total_pages,
        total_text_lines=total_text_lines,
        total_tables=total_tables,
        extraction_method=self.__class__.__name__,
        processing_time=processing_time
      )
      
    except (OSError, IOError) as e:
      raise PDFProcessingError(f"Failed to process PDF: {e}") from e 