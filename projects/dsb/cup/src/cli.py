"""
Command-line interface for PDF processing tools.
"""

import typer
from rich.console import Console
from rich.panel import Panel

from .app import PDFProcessor
from .core.config import Config
from .core.exceptions import PDFProcessingError, ConfigurationError

app = typer.Typer(help="PDF Processing Tools")
console = Console()


@app.command()
def extract(
  pdf_path: str = typer.Argument(..., help="Path to the PDF file"),
  output_format: str = typer.Option("csv", "--format", "-f", help="Output format: csv, txt, json, or all"),
  output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory"),
  extract_tables: bool = typer.Option(True, "--tables/--no-tables", help="Extract tables from PDF"),
  show_preview: bool = typer.Option(False, "--preview", help="Show preview of extracted data"),
  use_ocr: bool = typer.Option(False, "--ocr", help="Use OCR for text extraction"),
  ref_wtm_x: float = typer.Option(None, "--ref-x", help="Reference X coordinate in WTM format for address lookup"),
  ref_wtm_y: float = typer.Option(None, "--ref-y", help="Reference Y coordinate in WTM format for address lookup"),
  verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
  """Extract text from PDF with optional OCR."""
  
  try:
    # Initialize processor with reference coordinates
    processor = PDFProcessor(
      use_ocr=use_ocr,
      ref_wtm_x=ref_wtm_x,
      ref_wtm_y=ref_wtm_y
    )
    
    # Configure output
    output_config = {
      "format": output_format,
      "output_dir": output_dir,
      "include_tables": extract_tables,
      "show_preview": show_preview
    }
    
    # Process PDF
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
  output_path: str = typer.Option(None, "--output", "-o", help="Output CSV file path"),
  openai_api_key: str = typer.Option(None, "--api-key", help="OpenAI API key"),
  model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="OpenAI model to use"),
  extract_tables: bool = typer.Option(True, "--tables/--no-tables", help="Extract tables from PDF"),
  show_preview: bool = typer.Option(False, "--preview", help="Show preview of extracted data"),
  ref_wtm_x: float = typer.Option(None, "--ref-x", help="Reference X coordinate in WTM format for address lookup"),
  ref_wtm_y: float = typer.Option(None, "--ref-y", help="Reference Y coordinate in WTM format for address lookup"),
  verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
  """Convert PDF to CSV using OCR and AI post-processing."""
  
  try:
    # Validate OpenAI API key
    api_key = openai_api_key or Config.get_openai_api_key()
    if not api_key:
      console.print("❌ OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key", style="red")
      raise typer.Exit(1)
    
    if not Config.validate_openai_key(api_key):
      console.print("❌ Invalid OpenAI API key format", style="red")
      raise typer.Exit(1)
    
    # Set output path
    if not output_path:
      from pathlib import Path
      pdf_name = Path(pdf_path).stem
      output_path = f"{pdf_name}_output.csv"
    
    # Initialize processor with OCR, AI, and reference coordinates
    processor = PDFProcessor(
      use_ocr=True, 
      openai_api_key=api_key,
      ref_wtm_x=ref_wtm_x,
      ref_wtm_y=ref_wtm_y
    )
    
    # Process with AI
    processor.process_with_ai(pdf_path, output_path)
    
    # Show summary
    console.print("\n📊 Conversion Summary:", style="blue")
    console.print(f"  • Output file: {output_path}")
    console.print(f"  • Model used: {model}")
    
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
    "• convert - Convert PDFs to CSV using OCR and AI\n\n"
    "[bold]Features:[/bold]\n"
    "• Direct text extraction (no OCR required)\n"
    "• OCR-based text extraction using Surya\n"
    "• Table detection and extraction\n"
    "• AI-powered post-processing with OpenAI\n"
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