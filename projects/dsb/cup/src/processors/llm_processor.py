"""LLM post-processing using OpenAI API."""

from __future__ import annotations

from typing import Any, final, override, cast

import csv
from io import StringIO

import polars as pl

from rich.console import Console

from .base import BaseProcessor
from ..core.types import ProcessingResult, CSVRow
from ..core.config import Config
from ..utils.openai_client import call_openai


console = Console()


@final
class LLMProcessor(BaseProcessor):
  """Post-process extraction results using OpenAI API."""

  def __init__(
    self,
    api_key: str,
    model: str | None = None,
    names_only: bool = False,
  ) -> None:
    super().__init__()
    self.api_key: str = api_key
    self.model: str = model or Config.DEFAULT_MODEL
    self.names_only: bool = names_only

  @override
  def validate_config(self) -> bool:
    if not self.api_key:
      return False
    return Config.validate_openai_key(self.api_key)

  @override
  def process(self, result: ProcessingResult) -> str:
    console.print(f"🤖 Post-processing with OpenAI ({self.model})...", style="yellow")

    system_prompt, user_prompt = self._build_prompts(result)

    try:
      csv_content = self._invoke_openai(system_prompt, user_prompt)

      csv_content = csv_content.strip()
      if csv_content.startswith("```csv"):
        csv_content = csv_content[7:]
      if csv_content.startswith("```"):
        csv_content = csv_content[3:]
      if csv_content.endswith("```"):
        csv_content = csv_content[:-3]
      csv_content = csv_content.strip()

      if self.names_only:
        csv_content = self._enforce_names_only_csv(csv_content)

      return csv_content.strip()
    except (ValueError, RuntimeError) as e:
      console.print(f"❌ Error during LLM post-processing: {e}", style="red")
      return self._fallback_csv_conversion(result)

  def _invoke_openai(self, system_prompt: str, user_prompt: str) -> str:
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt},
    ]
    return call_openai(
      model=self.model,
      api_key=self.api_key,
      messages=messages,
      response_format=None,
    )

  def _prepare_text_content(self, result: ProcessingResult) -> str:
    content: list[str] = []
    for page in result.pages:
      content.append(f"Page {page.page}:")
      for line in page.text_lines:
        content.append(f"  {line.text}")
      content.append("")
    return "\n".join(content)

  def _prepare_table_content(self, result: ProcessingResult) -> str:
    all_tables = []
    for page in result.pages:
      all_tables.extend(page.tables)
    if not all_tables:
      return "No tables detected"
    content: list[str] = []
    for table in all_tables:
      content.append(f"Table on page {table.page}:")
      for row in table.rows:
        row_content = [str(value) for _, value in row.items()]
        content.append("  | ".join(row_content))
      content.append("")
    return "\n".join(content)

  def _build_prompts(self, result: ProcessingResult) -> tuple[str, str]:
    text_content = self._prepare_text_content(result)
    table_content = self._prepare_table_content(result)

    if self.names_only:
      system_prompt = (
        "You extract only restaurant or merchant names from Korean expense documents.\n"
        "Return a CSV with a single header 'restaurant_name'.\n"
        "Each row must contain exactly one unique, cleaned name (no amounts, dates, or extra columns)."
      )
      user_prompt = (
        "From the following OCR/text extraction, list unique restaurant or store names only.\n"
        "Normalize whitespace, remove surrounding quotes, and deduplicate.\n\n"
        f"TEXT CONTENT:\n{text_content}\n\nTABLE CONTENT:\n{table_content}\n\n"
        "Remember: output raw CSV with header 'restaurant_name' and no markdown."
      )
      return system_prompt, user_prompt

    system_prompt = (
      "You are an expert at fixing extraction errors and structuring data for CSV output.\n"
      "Your task is to:\n"
      "1. Fix any errors in the extracted text\n"
      "2. Identify and structure tabular data if present\n"
      "3. Return a clean, structured CSV format\n\n"
      "Guidelines:\n"
      "- Fix common OCR or extraction errors (0/O, 1/l, 5/S, etc.)\n"
      "- Preserve the original meaning and context\n"
      "- Structure data in a logical CSV format\n"
      "- Use appropriate headers\n"
      "- Handle missing or corrupted data gracefully\n\n"
      "IMPORTANT: Return ONLY the raw CSV data without any markdown formatting, code blocks, or explanations."
    )

    user_prompt = (
      "Please process this extracted data and return clean CSV:\n\n"
      f"TEXT CONTENT:\n{text_content}\n\n"
      f"TABLE CONTENT:\n{table_content}\n\n"
      "CRITICAL: Return ONLY the raw CSV data. Do not include any markdown formatting, code blocks, or explanations."
    )

    return system_prompt, user_prompt

  def _enforce_names_only_csv(self, csv_text: str) -> str:
    """Force CSV to contain a single 'restaurant_name' column."""
    cleaned = csv_text.strip()
    if not cleaned:
      return "restaurant_name\n"

    # Remove stray markdown fences if any remain
    cleaned = cleaned.replace("`", "")

    reader = csv.reader(StringIO(cleaned))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
      return "restaurant_name\n"

    keywords = ["restaurant", "store", "가맹점", "상호", "매장", "place"]
    header_row = rows[0]
    has_header = any(
      any(keyword in cell.strip().lower() for keyword in keywords)
      for cell in header_row
    )

    data_rows = rows[1:] if has_header else rows
    target_idx = 0
    if has_header:
      for idx, cell in enumerate(header_row):
        col_lower = cell.strip().lower()
        if any(keyword in col_lower for keyword in keywords):
          target_idx = idx
          break

    names: list[str] = []
    for row in data_rows:
      if not row:
        continue
      if target_idx >= len(row):
        continue
      name = row[target_idx].strip().strip('"')
      if name and name not in names:
        names.append(name)

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["restaurant_name"])
    for name in names:
      writer.writerow([name])

    return output.getvalue()

  def _fallback_csv_conversion(self, result: ProcessingResult) -> str:
    console.print("⚠️  Using fallback CSV conversion", style="yellow")
    csv_rows: list[CSVRow] = []
    for page in result.pages:
      for i, line in enumerate(page.text_lines, 1):
        csv_rows.append(
          CSVRow(
            page=page.page,
            line=i,
            text=line.text,
            confidence=line.confidence or 0.0,
            nearest_address=getattr(line, "nearest_address", None),
            x=getattr(line, "x", None),
            y=getattr(line, "y", None),
          )
        )
    rows_data = [row.model_dump() for row in csv_rows]
    df = pl.DataFrame(rows_data)
    return df.write_csv()
