"""
Command-line interface for PDF processing tools.
"""

from typing import Optional
import os
import typer
from rich.console import Console
from rich.panel import Panel

from .app import PDFProcessor
from .core.exceptions import PDFProcessingError
from .processors import LLMProcessor

app = typer.Typer(help="PDF Processing Tools")
console = Console()


@app.command()
def extract(
  pdf_path: str = typer.Argument(..., help="Path to the PDF file"),
  output_format: str = typer.Option("csv", "--format", "-f", help="Output format: csv, txt, json, or all"),
  output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
  extract_tables: bool = typer.Option(True, "--tables/--no-tables", help="Extract tables from PDF"),
  show_preview: bool = typer.Option(False, "--preview", help="Show preview of extracted data"),
  use_ocr: bool = typer.Option(False, "--ocr", help="Use OCR for text extraction"),
  ref_wtm_x: Optional[float] = typer.Option(None, "--ref-x", help="Reference X coordinate in WTM format for address lookup"),
  ref_wtm_y: Optional[float] = typer.Option(None, "--ref-y", help="Reference Y coordinate in WTM format for address lookup"),
  verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
  """Extract text from PDF with optional OCR."""
  
  try:
    processor = PDFProcessor(use_ocr=use_ocr, ref_wtm_x=ref_wtm_x, ref_wtm_y=ref_wtm_y)
    output_config = {
      "format": output_format,
      "output_dir": output_dir,
      "include_tables": extract_tables,
      "show_preview": show_preview,
    }
    processor.process_pdf(pdf_path, output_config)
    
  except PDFProcessingError as e:
    console.print(f"❌ Error: {e}", style="red")
    if verbose:
      import traceback
      console.print(traceback.format_exc(), style="red")
    raise typer.Exit(1)
  except (OSError, IOError, ValueError) as e:
    console.print(f"❌ Unexpected error: {e}", style="red")
    if verbose:
      import traceback
      console.print(traceback.format_exc(), style="red")
    raise typer.Exit(1)


@app.command()
def convert(
  pdf_path: str = typer.Argument(..., help="Path to the PDF file"),
  output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output CSV file path"),
  api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenRouter API key"),
  model: Optional[str] = typer.Option(None, "--model", "-m", help="OpenRouter model (default: google/gemini-2.5-pro)"),
  extract_tables: bool = typer.Option(True, "--tables/--no-tables", help="Extract tables from PDF"),
  show_preview: bool = typer.Option(False, "--preview", help="Show preview of extracted data"),
  ref_wtm_x: Optional[float] = typer.Option(None, "--ref-x", help="Reference X coordinate in WTM format for address lookup"),
  ref_wtm_y: Optional[float] = typer.Option(None, "--ref-y", help="Reference Y coordinate in WTM format for address lookup"),
  verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
  """Convert PDF to CSV using OCR and AI post-processing."""
  
  resolved_api_key = api_key or os.getenv("OPENROUTER_KEY")
  if not resolved_api_key:
    console.print("❌ API key not provided. Use --api-key or set OPENROUTER_KEY", style="red")
    raise typer.Exit(1)
  
  # Set output path
  if not output_path:
    from pathlib import Path
    pdf_name = Path(pdf_path).stem
    output_path = f"{pdf_name}_output.csv"
  
  processor = PDFProcessor(use_ocr=True, ref_wtm_x=ref_wtm_x, ref_wtm_y=ref_wtm_y)
  processor.ai_processor = LLMProcessor(resolved_api_key, model=model)
  
  # Process with AI
  processor.process_with_ai(pdf_path, output_path)
  
  # Show summary
  console.print("\n📊 Conversion Summary:", style="blue")
  console.print(f"  • Output file: {output_path}")
  console.print(f"  • Provider: openrouter")
  console.print(f"  • Model: {processor.ai_processor.model}")


@app.command()
def extract_llm(
  pdf_path: str = typer.Argument(..., help="Path to the PDF file"),
  output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output CSV file path"),
  api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenRouter API key"),
  model: Optional[str] = typer.Option(None, "--model", "-m", help="OpenRouter model (default: google/gemini-2.5-pro)"),
  use_ocr: bool = typer.Option(False, "--ocr", help="Use OCR for text extraction (otherwise direct text)"),
  show_preview: bool = typer.Option(False, "--preview", help="Show preview of extracted data"),
  ref_wtm_x: Optional[float] = typer.Option(None, "--ref-x", help="Reference X coordinate in WTM format for address lookup"),
  ref_wtm_y: Optional[float] = typer.Option(None, "--ref-y", help="Reference Y coordinate in WTM format for address lookup"),
  verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
  """Direct or OCR extraction, then LLM post-processing to CSV."""
  try:
    resolved_api_key = api_key or os.getenv("OPENROUTER_KEY")
    if not resolved_api_key:
      console.print("❌ API key not provided. Use --api-key or set OPENROUTER_KEY", style="red")
      raise typer.Exit(1)

    # Set output path
    if not output_path:
      from pathlib import Path
      pdf_name = Path(pdf_path).stem
      output_path = f"{pdf_name}_llm.csv"

    processor = PDFProcessor(use_ocr=use_ocr, ref_wtm_x=ref_wtm_x, ref_wtm_y=ref_wtm_y)
    processor.ai_processor = LLMProcessor(resolved_api_key, model=model)

    # Extract data
    text_pages = processor.extractor.extract_text(pdf_path)
    table_pages: list = []
    combined_pages = processor._combine_pages(text_pages, table_pages)
    result = processor._create_processing_result(pdf_path, combined_pages)

    # Post-process with LLM to CSV
    csv_content = processor.ai_processor.process(result) if processor.ai_processor else ""
    with open(output_path, "w", encoding="utf-8") as f:
      f.write(csv_content)
    console.print(f"✅ LLM-processed CSV saved to: {output_path}", style="green")

  except PDFProcessingError as e:
    console.print(f"❌ Error: {e}", style="red")
    if verbose:
      import traceback
      console.print(traceback.format_exc(), style="red")
    raise typer.Exit(1)
  except (OSError, IOError, ValueError) as e:
    console.print(f"❌ Unexpected error: {e}", style="red")
    if verbose:
      import traceback
      console.print(traceback.format_exc(), style="red")
    raise typer.Exit(1)


@app.command()
def info():
  """Show information about the tools."""
  console.print(Panel.fit(
    "[bold blue]PDF Processing Tools[/bold blue]\n\n"
    "A collection of tools for extracting and converting PDF content.\n\n"
    "[bold]Available Commands:[/bold]\n"
    "• extract - Extract text from PDFs (with or without OCR)\n"
    "• convert - Convert PDFs to CSV using OCR and AI (OpenRouter)\n\n"
    "[bold]Features:[/bold]\n"
    "• Direct text extraction (no OCR required)\n"
    "• OCR-based text extraction using Surya\n"
    "• Table detection and extraction\n"
    "• AI-powered post-processing with OpenRouter (OpenAI-compatible)\n"
    "• Multiple output formats (CSV, TXT, JSON)\n"
    "• Fast processing and high accuracy\n\n"
    "[bold]When to use extract:[/bold]\n"
    "• PDFs with text layers (most modern PDFs)\n"
    "• Scanned documents converted to text\n"
    "• Forms and documents with embedded text\n\n"
    "[bold]When to use convert:[/bold]\n"
    "• Pure image-based PDFs\n"
    "• Scanned documents without OCR\n"
    "• Handwritten documents\n"
    "• When you need AI-powered error correction",
    title="Tool Information"
  ))


if __name__ == "__main__":
  app() 