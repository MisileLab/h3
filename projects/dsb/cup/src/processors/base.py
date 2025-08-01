"""
Base processor class for AI post-processing.
"""

from abc import ABC, abstractmethod
from typing import final

from ..core.types import ProcessingResult
from ..core.exceptions import PDFProcessingError


class BaseProcessor(ABC):
  """Base class for AI post-processors."""
  
  def __init__(self) -> None:
    """Initialize the processor."""
    pass
  
  @abstractmethod
  def process(self, result: ProcessingResult) -> str:
    """Process the extraction result and return formatted output."""
    pass
  
  @abstractmethod
  def validate_config(self) -> bool:
    """Validate the processor configuration."""
    pass 