"""
Direct text extraction using pypdf (no OCR required).
"""

from typing import override, final
import pypdf
import tabula

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .base import BaseExtractor
from ..core.types import PageData, TextLine, TableData
from ..core.exceptions import PDFReadError

console = Console()


@final
class DirectTextExtractor(BaseExtractor):
  """Extract text directly from PDFs without OCR."""
  
  def __init__(self) -> None:
    """Initialize the direct text extractor."""
    super().__init__()
    console.print("📄 Initializing direct text extractor...", style="blue")
  
  @override
  def extract_text(self, pdf_path: str) -> list[PageData]:
    """Extract text from PDF using direct text extraction."""
    console.print(f"📄 Processing PDF: {pdf_path}", style="blue")

    pages_data: list[PageData] = []

    try:
      with open(pdf_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        total_pages = len(pdf_reader.pages)

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
              ) as progress:
          task = progress.add_task("Extracting text from pages...", total=total_pages)

          for page_num in range(total_pages):
            progress.update(task, description=f"Processing page {page_num + 1}/{total_pages}")

            page = pdf_reader.pages[page_num]
            text = page.extract_text()

            # Convert text to TextLine objects
            text_lines: list[TextLine] = []
            if text:
              for line_num, line in enumerate(text.split("\n"), 1):
                if line := line.strip():
                  text_lines.append(TextLine(
                    text=line,
                    line_number=line_num,
                    page=page_num + 1,
                    confidence=None,
                    bbox=None
                  ))

            page_data = PageData(
              page=page_num + 1,
              text_lines=text_lines,
              tables=[],  # Direct extraction doesn't handle tables
              total_lines=len(text_lines),
              raw_text=text or ""
            )
            pages_data.append(page_data)

            progress.advance(task)

        console.print(f"✅ Extracted text from {total_pages} pages", style="green")

    except (OSError, IOError, ValueError) as e:
      raise PDFReadError(f"Error reading PDF: {e}") from e

    return pages_data
  
  @override
  def extract_tables(self, pdf_path: str) -> list[PageData]:
    """Extract tables using tabula-py."""
    console.print(f"📊 Extracting tables from PDF: {pdf_path}", style="blue")
    
    pages_data: list[PageData] = []
    
    try:
      # Extract all tables from the PDF
      tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True) # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
      if isinstance(tables, dict):
        raise ValueError("Tabula returned a dictionary instead of a list of tables")
      
      with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
      ) as progress:
        task = progress.add_task("Processing tables...", total=len(tables))
        
        # Group tables by page (simplified - tabula doesn't provide page info easily)
        page_tables: list[TableData] = []
        for table_idx, table in enumerate(tables):
          progress.update(task, description=f"Processing table {table_idx + 1}/{len(tables)}")
          
          # Convert table to list of dictionaries
          table_data: list[dict[str, str]] = []
          for _, row in table.iterrows():
            row_dict: dict[str, str] = {}
            for col_idx, value in enumerate(row): # pyright: ignore[reportUnknownVariableType]
              col_name = table.columns[col_idx] if col_idx < len(table.columns) else f"Column_{col_idx}" # pyright: ignore[reportUnknownVariableType]
              if not isinstance(col_name, str):
                raise ValueError(f"Column name is not a string: {col_name}")
              row_dict[col_name] = str(value) if value is not None else "" # pyright: ignore[reportUnknownArgumentType]
            table_data.append(row_dict)
          
          table_result = TableData(
            table_index=table_idx + 1,
            page=1,  # Simplified - assume all tables on first page
            rows=table_data,
            columns=list(table.columns),
            shape=table.shape
          )
          page_tables.append(table_result)
          
          progress.advance(task)
        
        # Create a single page with all tables
        if page_tables:
          page_data = PageData(
            page=1,
            text_lines=[],
            tables=page_tables,
            total_lines=0,
            raw_text=""
          )
          pages_data.append(page_data)
        
        console.print(f"✅ Extracted {len(tables)} tables", style="green")
        
    except (OSError, IOError, ValueError) as e:
      console.print(f"❌ Error extracting tables: {e}", style="red")
      console.print("   Continuing without table extraction.", style="yellow")
    
    return pages_data 