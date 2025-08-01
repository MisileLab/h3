"""
OpenAI-powered post-processing for PDF extraction results.
"""

from typing import Any, override, final
import polars as pl
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from rich.console import Console

from .base import BaseProcessor
from ..core.types import ProcessingResult, CSVRow
from ..core.exceptions import PDFProcessingError, ConfigurationError, DependencyError
from ..core.config import Config

console = Console()


@final
class OpenAIProcessor(BaseProcessor):
  """Post-process OCR results using OpenAI to fix errors and structure data."""
  
  def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
    """Initialize the OpenAI processor."""
    super().__init__()
    self.api_key: str = api_key
    self.model: str = model
    
    self.llm: ChatOpenAI = ChatOpenAI(
      api_key=api_key,
      model=model,
      temperature=Config.DEFAULT_TEMPERATURE
    )
    self.HumanMessage: type[HumanMessage] = HumanMessage
    self.SystemMessage: type[SystemMessage] = SystemMessage
  
  @override
  def validate_config(self) -> bool:
    """Validate the OpenAI configuration."""
    if not self.api_key:
      return False
    return Config.validate_openai_key(self.api_key)
  
  @override
  def process(self, result: ProcessingResult) -> str:
    """Post-process OCR results using OpenAI."""
    console.print("🤖 Post-processing with OpenAI...", style="yellow")
    
    # Prepare data for OpenAI
    text_content = self._prepare_text_content(result)
    table_content = self._prepare_table_content(result)
    
    system_prompt = """You are an expert at fixing OCR errors and structuring data for CSV output. 
Your task is to:
1. Fix any OCR errors in the extracted text
2. Identify and structure tabular data
3. Return a clean, structured CSV format

Guidelines:
- Fix common OCR errors (0/O, 1/l, 5/S, etc.)
- Preserve the original meaning and context
- Structure data in a logical CSV format
- Use appropriate headers
- Handle missing or corrupted data gracefully
- If multiple tables are found, combine them logically or create separate sections

IMPORTANT: Return ONLY the raw CSV data without any markdown formatting, code blocks, or explanations. 
Do not include ```csv or ``` markers. Start directly with the header row and end with the last data row."""
    
    user_prompt = f"""Please process this OCR data and return clean CSV:

TEXT CONTENT:
{text_content}

TABLE CONTENT:
{table_content}

CRITICAL: Return ONLY the raw CSV data. Do not include any markdown formatting, code blocks, or explanations.
Start with the header row and end with the last data row. No ```csv or ``` markers."""
    
    try:
      response = self.llm.invoke([
        self.SystemMessage(content=system_prompt),
        self.HumanMessage(content=user_prompt)
      ])
      
      # Clean up any markdown formatting that might slip through
      csv_content = response.content.strip()
      
      # Remove markdown code block markers
      if csv_content.startswith("```csv"):
        csv_content = csv_content[7:]  # Remove ```csv
      elif csv_content.startswith("```"):
        csv_content = csv_content[3:]   # Remove ```
      
      if csv_content.endswith("```"):
        csv_content = csv_content[:-3]  # Remove trailing ```
      
      return csv_content.strip()
      
    except (OSError, IOError, ValueError, ConnectionError) as e:
      console.print(f"❌ Error in OpenAI post-processing: {e}", style="red")
      # Fallback to basic CSV conversion
      return self._fallback_csv_conversion(result)
  
  def _prepare_text_content(self, result: ProcessingResult) -> str:
    """Prepare text content for OpenAI processing."""
    content = []
    for page in result.pages:
      content.append(f"Page {page.page}:")
      for line in page.text_lines:
        content.append(f"  {line.text}")
      content.append("")
    return "\n".join(content)
  
  def _prepare_table_content(self, result: ProcessingResult) -> str:
    """Prepare table content for OpenAI processing."""
    all_tables = []
    for page in result.pages:
      all_tables.extend(page.tables)
    
    if not all_tables:
      return "No tables detected"
    
    content = []
    for table in all_tables:
      content.append(f"Table on page {table.page}:")
      
      # Convert table rows to text representation
      for row in table.rows:
        row_content = []
        for col_name, value in row.items():
          row_content.append(str(value))
        content.append("  | ".join(row_content))
      
      content.append("")
    
    return "\n".join(content)
  
  def _fallback_csv_conversion(self, result: ProcessingResult) -> str:
    """Fallback CSV conversion when OpenAI fails."""
    console.print("⚠️  Using fallback CSV conversion", style="yellow")
    
    # Simple CSV conversion using Pydantic models
    csv_rows = []
    
    for page in result.pages:
      page_num = page.page
      for i, line in enumerate(page.text_lines, 1):
        csv_row = CSVRow(
          page=page_num,
          line=i,
          text=line.text,
          confidence=line.confidence or 0.0
        )
        csv_rows.append(csv_row)
    
    # Convert Pydantic models to dictionaries for Polars
    rows_data = [row.model_dump() for row in csv_rows]
    df = pl.DataFrame(rows_data)
    return df.write_csv() 