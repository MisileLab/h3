"""
JSON output formatter.
"""

import json
from pathlib import Path
from typing import Any, Dict, override, final

from rich.console import Console

from .base import BaseOutputFormatter
from ..core.types import ProcessingResult, TextLine, TableData, PageData
from ..core.exceptions import OutputError

console = Console()


@final
class JSONFormatter(BaseOutputFormatter):
  """Format processing results as JSON."""
  
  @override
  def format(self, result: ProcessingResult) -> str:
    """Format the processing result as JSON."""
    # Convert Pydantic models to dictionaries using model_dump()
    result_dict = result.model_dump()
    
    return json.dumps(result_dict, ensure_ascii=False, indent=2)
  
  @override
  def save(self, result: ProcessingResult, output_path: str) -> None:
    """Save the processing result as JSON."""
    try:
      self.ensure_directory(output_path)
      
      json_content = self.format(result)
      with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_content)
      
      console.print(f"✅ JSON saved to: {output_path}", style="green")
      
    except (OSError, IOError, ValueError) as e:
      raise OutputError(f"Error saving JSON: {e}") from e 