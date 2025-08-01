"""
Custom exceptions for PDF processing tools.
"""


class PDFProcessingError(Exception):
  """Base exception for PDF processing errors."""
  pass


class PDFReadError(PDFProcessingError):
  """Raised when there's an error reading the PDF file."""
  pass


class TextExtractionError(PDFProcessingError):
  """Raised when text extraction fails."""
  pass


class TableExtractionError(PDFProcessingError):
  """Raised when table extraction fails."""
  pass


class OutputError(PDFProcessingError):
  """Raised when there's an error saving output files."""
  pass


class ConfigurationError(PDFProcessingError):
  """Raised when there's a configuration error."""
  pass


class DependencyError(PDFProcessingError):
  """Raised when required dependencies are missing."""
  pass 