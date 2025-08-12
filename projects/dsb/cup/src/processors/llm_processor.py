"""LLM post-processing using OpenRouter (OpenAI-compatible HTTP API)."""

from __future__ import annotations

from typing import Any, final, override, cast

import httpx
import polars as pl

from rich.console import Console

from .base import BaseProcessor
from ..core.types import ProcessingResult, CSVRow
from ..core.config import Config


console = Console()


@final
class LLMProcessor(BaseProcessor):
  """Post-process extraction results using OpenRouter."""

  def __init__(
    self,
    api_key: str,
    model: str | None = None,
  ) -> None:
    super().__init__()
    self.api_key: str = api_key
    self.model: str = model or Config.DEFAULT_OPENROUTER_MODEL

  @override
  def validate_config(self) -> bool:
    if not self.api_key:
      return False
    return Config.validate_openrouter_key(self.api_key)

  @override
  def process(self, result: ProcessingResult) -> str:
    console.print(f"🤖 Post-processing with OpenRouter ({self.model})...", style="yellow")

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

    text_content = self._prepare_text_content(result)
    table_content = self._prepare_table_content(result)

    user_prompt = (
      "Please process this extracted data and return clean CSV:\n\n"
      f"TEXT CONTENT:\n{text_content}\n\n"
      f"TABLE CONTENT:\n{table_content}\n\n"
      "CRITICAL: Return ONLY the raw CSV data. Do not include any markdown formatting, code blocks, or explanations."
    )

    try:
      csv_content = self._invoke_openrouter(system_prompt, user_prompt)

      csv_content = csv_content.strip()
      if csv_content.startswith("```csv"):
        csv_content = csv_content[7:]
      if csv_content.startswith("```"):
        csv_content = csv_content[3:]
      if csv_content.endswith("```"):
        csv_content = csv_content[:-3]
      return csv_content.strip()
    except httpx.HTTPError as e:
      console.print(f"❌ HTTP error during LLM post-processing: {e}", style="red")
      return self._fallback_csv_conversion(result)
    except (ValueError, RuntimeError) as e:
      console.print(f"❌ Error during LLM post-processing: {e}", style="red")
      return self._fallback_csv_conversion(result)

  def _invoke_openrouter(self, system_prompt: str, user_prompt: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
      "Authorization": f"Bearer {self.api_key}",
      "Content-Type": "application/json",
      # Optional but recommended by OpenRouter
      "HTTP-Referer": "https://github.com/",  # Replace with your app/site
      "X-Title": "cup-pdf-tools",
    }
    payload: dict[str, Any] = {
      "model": self.model,
      "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
      ],
      "temperature": Config.DEFAULT_TEMPERATURE,
    }
    with httpx.Client(timeout=60) as client:
      resp = client.post(url, headers=headers, json=payload)
      resp.raise_for_status()
      data = cast(dict[str, Any], resp.json())
      try:
        content = cast(str, data["choices"][0]["message"]["content"])
        return content
      except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected OpenRouter response schema: {e}")

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


