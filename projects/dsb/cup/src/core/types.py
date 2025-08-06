"""
Type definitions for PDF processing tools.
"""

from typing import Any, TypedDict
from pydantic import BaseModel, Field, ConfigDict


class TextLine(BaseModel):
  """Represents a single line of extracted text."""
  model_config = ConfigDict(validate_assignment=True)
  
  text: str = Field(..., description="The extracted text content")
  line_number: int = Field(..., description="Line number on the page")
  page: int = Field(..., description="Page number")
  confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence score for OCR")
  bbox: tuple[float, float, float, float] | None = Field(None, description="Bounding box coordinates")
  nearest_address: str | None = Field(None, description="Nearest address found")
  x: float | None = Field(None, description="X coordinate (longitude)")
  y: float | None = Field(None, description="Y coordinate (latitude)")


class TableData(BaseModel):
  """Represents extracted table data."""
  model_config = ConfigDict(validate_assignment=True)
  
  table_index: int = Field(..., ge=0, description="Index of the table on the page")
  page: int = Field(..., ge=1, description="Page number")
  rows: list[dict[str, Any]] = Field(default_factory=list, description="Table rows as list of dictionaries") # pyright: ignore[reportExplicitAny]
  columns: list[str] = Field(default_factory=list, description="Column headers")
  shape: tuple[int, int] = Field(..., description="Table dimensions (rows, columns)")
  address_column: str | None = Field(None, description="Column name containing address information")


class PageData(BaseModel):
  """Represents data extracted from a single page."""
  model_config = ConfigDict(validate_assignment=True)
  
  page: int = Field(..., ge=1, description="Page number")
  text_lines: list[TextLine] = Field(default_factory=list, description="Extracted text lines")
  tables: list[TableData] = Field(default_factory=list, description="Extracted tables")
  total_lines: int = Field(..., ge=0, description="Total number of text lines")
  raw_text: str = Field(..., description="Raw extracted text")


class ProcessingResult(BaseModel):
  """Represents the complete result of PDF processing."""
  model_config = ConfigDict(validate_assignment=True)
  
  pdf_path: str = Field(..., description="Path to the processed PDF file")
  pages: list[PageData] = Field(default_factory=list, description="Processed pages")
  total_pages: int = Field(..., ge=0, description="Total number of pages")
  total_text_lines: int = Field(..., ge=0, description="Total number of text lines")
  total_tables: int = Field(..., ge=0, description="Total number of tables")
  extraction_method: str = Field(..., description="Method used for extraction")
  processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")


class CSVRow(BaseModel):
  """Represents a single row in CSV output."""
  model_config = ConfigDict(validate_assignment=True)
  
  page: int = Field(..., ge=1, description="Page number")
  line: int = Field(..., ge=1, description="Line number")
  text: str = Field(..., description="Text content")
  confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
  nearest_address: str | None = Field(None, description="Nearest address found")
  x: float | None = Field(None, description="X coordinate (longitude)")
  y: float | None = Field(None, description="Y coordinate (latitude)")


class OutputConfig(TypedDict, total=False):
  """Configuration for output formatting."""
  format: str  # 'csv', 'txt', 'json', 'all'
  output_dir: str
  include_tables: bool
  show_preview: bool
  verbose: bool 