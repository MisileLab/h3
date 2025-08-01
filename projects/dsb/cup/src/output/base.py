"""
Base output formatter class.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from ..core.types import ProcessingResult


class BaseOutputFormatter(ABC):
  """Base class for output formatters."""
  
  def __init__(self) -> None:
    """Initialize the formatter."""
    pass
  
  @abstractmethod
  def format(self, result: ProcessingResult) -> str:
    """Format the processing result."""
    pass
  
  @abstractmethod
  def save(self, result: ProcessingResult, output_path: str) -> None:
    """Save the formatted result to a file."""
    pass
  
  def ensure_directory(self, output_path: str) -> None:
    """Ensure the output directory exists."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True) 