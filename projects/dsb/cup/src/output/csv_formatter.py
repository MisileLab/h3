"""
CSV output formatter.
"""

import polars as pl
from pathlib import Path
from typing import override, final

from rich.console import Console

from .base import BaseOutputFormatter
from ..core.types import ProcessingResult, CSVRow
from ..core.exceptions import OutputError

console = Console()


@final
class CSVFormatter(BaseOutputFormatter):
  """Format processing results as CSV."""
  
  @override
  def format(self, result: ProcessingResult) -> str:
    """Format the processing result as CSV."""
    csv_rows = []

    for page in result.pages:
      page_num = page.page
      for line in page.text_lines:
        csv_row = CSVRow(
          page=page_num,
          line=line.line_number,
          text=line.text,
          confidence=line.confidence or 0.0
        )
        csv_rows.append(csv_row)
    
    # Convert Pydantic models to dictionaries for Polars
    rows_data = [row.model_dump() for row in csv_rows]
    df = pl.DataFrame(rows_data)
    return df.write_csv()
  
  @override
  def save(self, result: ProcessingResult, output_path: str) -> None:
    """Save the processing result as CSV."""
    try:
      self.ensure_directory(output_path)
      
      # Save text data
      csv_content = self.format(result)
      with open(output_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
      
      console.print(f"✅ CSV saved to: {output_path}", style="green")
      
      # Save tables if they exist
      self._save_tables(result, output_path)
      
    except (OSError, IOError, ValueError) as e:
      raise OutputError(f"Error saving CSV: {e}") from e
  
  def _save_tables(self, result: ProcessingResult, base_path: str) -> None:
    """Save tables to separate CSV files."""
    all_tables = []
    for page in result.pages:
      all_tables.extend(page.tables)
    
    if not all_tables:
      return
    
    base_path_obj = Path(base_path)
    base_name = base_path_obj.stem
    base_dir = base_path_obj.parent
    
    # For single table, save in normal table format
    if len(all_tables) == 1:
      table = all_tables[0]
      df = pl.DataFrame(table.rows)
      table_path = base_dir / f"{base_name}_table.csv"
      df.write_csv(table_path)
      console.print(f"✅ Table CSV saved to: {table_path}", style="green")
    else:
      # For multiple tables, save each to separate files
      for table in all_tables:
        table_idx = table.table_index
        table_path = base_dir / f"{base_name}_table_{table_idx}.csv"
        df = pl.DataFrame(table.rows)
        df.write_csv(table_path)
        console.print(f"✅ Table {table_idx} CSV saved to: {table_path}", style="green")
      
      # Also create a combined file with table identification
      combined_rows = []
      for table in all_tables:
        table_idx = table.table_index
        for row in table.rows:
          row_with_table = {"Table": f"Table_{table_idx}", **row}
          combined_rows.append(row_with_table)
      
      combined_df = pl.DataFrame(combined_rows)
      combined_path = base_dir / f"{base_name}_combined.csv"
      combined_df.write_csv(combined_path)
      console.print(f"✅ Combined tables CSV saved to: {combined_path}", style="green") 