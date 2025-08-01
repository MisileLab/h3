"""
Tests for core functionality.
"""

import pytest
from pathlib import Path

from src.core.config import Config
from src.core.types import TextLine, TableData, PageData, ProcessingResult, CSVRow
from src.core.exceptions import PDFProcessingError


class TestConfig:
  """Test configuration management."""
  
  def test_validate_openai_key(self):
    """Test OpenAI API key validation."""
    # Valid key
    assert Config.validate_openai_key("sk-1234567890abcdef1234567890abcdef1234567890abcdef")
    
    # Invalid keys
    assert not Config.validate_openai_key("invalid-key")
    assert not Config.validate_openai_key("sk-123")
    assert not Config.validate_openai_key("")
  
  def test_get_output_dir(self):
    """Test output directory generation."""
    # With custom output dir
    assert Config.get_output_dir("test.pdf", "/custom/path") == "/custom/path"
    
    # Without custom output dir
    result = Config.get_output_dir("test.pdf")
    assert "test_extract" in result
  
  def test_get_output_path(self):
    """Test output path generation."""
    path = Config.get_output_path("test.pdf", "/output", "csv")
    # Handle path separator differences on different platforms
    expected = str(Path("/output/test_csv.csv"))
    assert path == expected


class TestTypes:
  """Test type definitions."""
  
  def test_text_line(self):
    """Test TextLine Pydantic model."""
    line = TextLine(
      text="Test text",
      line_number=1,
      page=1,
      confidence=0.95
    )
    
    assert line.text == "Test text"
    assert line.line_number == 1
    assert line.page == 1
    assert line.confidence == 0.95
  

  
  def test_table_data(self):
    """Test TableData Pydantic model."""
    table = TableData(
      table_index=1,
      page=1,
      rows=[{"col1": "val1", "col2": "val2"}],
      columns=["col1", "col2"],
      shape=(1, 2)
    )
    
    assert table.table_index == 1
    assert table.page == 1
    assert len(table.rows) == 1
    assert len(table.columns) == 2
    assert table.shape == (1, 2)
  

  
  def test_page_data(self):
    """Test PageData Pydantic model."""
    text_line = TextLine(text="Test", line_number=1, page=1)
    table = TableData(
      table_index=1,
      page=1,
      rows=[],
      columns=[],
      shape=(0, 0)
    )
    
    page = PageData(
      page=1,
      text_lines=[text_line],
      tables=[table],
      total_lines=1,
      raw_text="Test"
    )
    
    assert page.page == 1
    assert len(page.text_lines) == 1
    assert len(page.tables) == 1
    assert page.total_lines == 1
    assert page.raw_text == "Test"
  

  
  def test_processing_result(self):
    """Test ProcessingResult Pydantic model."""
    text_line = TextLine(text="Test", line_number=1, page=1)
    page = PageData(
      page=1,
      text_lines=[text_line],
      tables=[],
      total_lines=1,
      raw_text="Test"
    )
    
    result = ProcessingResult(
      pdf_path="test.pdf",
      pages=[page],
      total_pages=1,
      total_text_lines=1,
      total_tables=0,
      extraction_method="TestExtractor",
      processing_time=1.0
    )
    
    assert result.pdf_path == "test.pdf"
    assert result.total_pages == 1
    assert result.total_text_lines == 1
    assert result.total_tables == 0
    assert result.extraction_method == "TestExtractor"
    assert result.processing_time == 1.0
  
  def test_csv_row(self):
    """Test CSVRow Pydantic model."""
    csv_row = CSVRow(
      page=1,
      line=1,
      text="Test text",
      confidence=0.95
    )
    
    assert csv_row.page == 1
    assert csv_row.line == 1
    assert csv_row.text == "Test text"
    assert csv_row.confidence == 0.95
    
    # Test default confidence
    csv_row_default = CSVRow(
      page=1,
      line=1,
      text="Test text"
    )
    assert csv_row_default.confidence == 0.0
  



class TestExceptions:
  """Test custom exceptions."""
  
  def test_pdf_processing_error(self):
    """Test PDFProcessingError."""
    error = PDFProcessingError("Test error")
    assert str(error) == "Test error"
  
  def test_exception_inheritance(self):
    """Test exception inheritance."""
    from src.core.exceptions import (
      PDFReadError, TextExtractionError, TableExtractionError,
      OutputError, ConfigurationError, DependencyError
    )
    
    # All should inherit from PDFProcessingError
    assert issubclass(PDFReadError, PDFProcessingError)
    assert issubclass(TextExtractionError, PDFProcessingError)
    assert issubclass(TableExtractionError, PDFProcessingError)
    assert issubclass(OutputError, PDFProcessingError)
    assert issubclass(ConfigurationError, PDFProcessingError)
    assert issubclass(DependencyError, PDFProcessingError) 