"""
Text output formatter.
"""

from pathlib import Path
from typing import override, final

from rich.console import Console

from .base import BaseOutputFormatter
from ..core.types import ProcessingResult
from ..core.exceptions import OutputError

console = Console()


@final
class TextFormatter(BaseOutputFormatter):
  """Format processing results as plain text."""
  
  @override
  def format(self, result: ProcessingResult) -> str:
    """Format the processing result as plain text."""
    lines = []
    
    for page in result.pages:
      lines.append(f"\n=== Page {page.page} ===\n")
      for line in page.text_lines:
        lines.append(line.text)
      lines.append("")
    
    return "\n".join(lines)
  
  @override
  def save(self, result: ProcessingResult, output_path: str) -> None:
    """Save the processing result as plain text."""
    try:
      self.ensure_directory(output_path)
      
      text_content = self.format(result)
      with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_content)
      
      console.print(f"✅ Text saved to: {output_path}", style="green")
      
    except (OSError, IOError, ValueError) as e:
      raise OutputError(f"Error saving text file: {e}") from e 