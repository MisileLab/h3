"""
Command-line interface for PDF processing tools.
"""

from typing import Optional
import os
from datetime import datetime
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel

from .app import PDFProcessor
from .core.exceptions import PDFProcessingError
from .processors import LLMProcessor
from .scrapers import DongjakScraper
from .db import init_database, PDFRepository

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
  store_db: bool = typer.Option(False, "--store-db", help="Store results in database"),
  db_path: str = typer.Option("./data.db", "--db-path", help="SQLite database path"),
  verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
  """Extract text from PDF with optional OCR."""

  try:
    processor = PDFProcessor(
      use_ocr=use_ocr,
      ref_wtm_x=ref_wtm_x,
      ref_wtm_y=ref_wtm_y,
      db_path=db_path if store_db else None,
      store_in_db=store_db
    )
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
  api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenAI API key"),
  model: Optional[str] = typer.Option(None, "--model", "-m", help="OpenAI model (default: gpt-4o-mini)"),
  extract_tables: bool = typer.Option(True, "--tables/--no-tables", help="Extract tables from PDF"),
  show_preview: bool = typer.Option(False, "--preview", help="Show preview of extracted data"),
  ref_wtm_x: Optional[float] = typer.Option(None, "--ref-x", help="Reference X coordinate in WTM format for address lookup"),
  ref_wtm_y: Optional[float] = typer.Option(None, "--ref-y", help="Reference Y coordinate in WTM format for address lookup"),
  verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
  """Convert PDF to CSV using OCR and AI post-processing."""

  resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
  if not resolved_api_key:
    console.print("❌ API key not provided. Use --api-key or set OPENAI_API_KEY", style="red")
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
  console.print(f"  • Provider: OpenAI")
  console.print(f"  • Model: {processor.ai_processor.model}")


@app.command()
def extract_llm(
  pdf_path: str = typer.Argument(..., help="Path to the PDF file"),
  output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output CSV file path"),
  api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenAI API key"),
  model: Optional[str] = typer.Option(None, "--model", "-m", help="OpenAI model (default: gpt-4o-mini)"),
  use_ocr: bool = typer.Option(False, "--ocr", help="Use OCR for text extraction (otherwise direct text)"),
  show_preview: bool = typer.Option(False, "--preview", help="Show preview of extracted data"),
  ref_wtm_x: Optional[float] = typer.Option(None, "--ref-x", help="Reference X coordinate in WTM format for address lookup"),
  ref_wtm_y: Optional[float] = typer.Option(None, "--ref-y", help="Reference Y coordinate in WTM format for address lookup"),
  verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
  """Direct or OCR extraction, then LLM post-processing to CSV."""
  try:
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
      console.print("❌ API key not provided. Use --api-key or set OPENAI_API_KEY", style="red")
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
def scrape(
  url: str = typer.Argument(
    "https://www.dongjak.go.kr/portal/bbs/B0000591/list.do?menuNo=200209",
    help="URL of the bulletin board to scrape"
  ),
  limit: int = typer.Option(0, "--limit", "-l", help="Limit number of PDFs to download (0 = no limit)"),
  use_ocr: bool = typer.Option(False, "--ocr", help="Use OCR for extraction"),
  skip_existing: bool = typer.Option(True, "--skip-existing", help="Skip already downloaded files"),
  db_path: str = typer.Option("./data.db", "--db-path", help="SQLite database path"),
  output_dir: str = typer.Option("./downloads/dongjak", "--output-dir", "-o", help="Download directory"),
  ref_wtm_x: Optional[float] = typer.Option(None, "--ref-x", help="Reference X coordinate in WTM format"),
  ref_wtm_y: Optional[float] = typer.Option(None, "--ref-y", help="Reference Y coordinate in WTM format"),
  verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
  """Scrape PDFs from Dongjak government website and process them."""
  try:
    # Initialize database
    init_database(db_path)
    repository = PDFRepository(db_path)

    # Initialize scraper
    scraper = DongjakScraper()
    console.print(f"[blue]Scraping PDFs from: {url}[/blue]")

    # Scrape PDF links
    pdf_links = scraper.scrape(url, limit=limit)

    if not pdf_links:
      console.print("[yellow]No PDFs found.[/yellow]")
      return

    console.print(f"[green]Found {len(pdf_links)} PDFs[/green]")

    # Process each PDF
    processed = 0
    skipped = 0
    failed = 0

    for idx, pdf_link in enumerate(pdf_links, 1):
      console.print(f"\n[blue]Processing {idx}/{len(pdf_links)}: {pdf_link.title or pdf_link.url}[/blue]")

      # Check if URL already processed (caching)
      if skip_existing and repository.url_exists(pdf_link.url):
        console.print("[yellow]Already processed, skipping...[/yellow]")
        skipped += 1
        continue

      # Add to scraping queue
      repository.add_to_scraping_queue(pdf_link.url, pdf_link.title)

      try:
        # Download PDF
        pdf_path = scraper.download_pdf(pdf_link, output_dir)
        repository.update_queue_status(pdf_link.url, "downloaded", pdf_path)

        # Process PDF
        processor = PDFProcessor(
          use_ocr=use_ocr,
          ref_wtm_x=ref_wtm_x,
          ref_wtm_y=ref_wtm_y,
          db_path=db_path,
          store_in_db=True
        )

        output_config = {
          "format": "csv",
          "output_dir": None,
          "include_tables": True,
          "show_preview": False,
          "source_url": pdf_link.url,
          "download_date": datetime.now(),
        }

        processor.process_pdf(pdf_path, output_config)
        repository.update_queue_status(pdf_link.url, "processed")
        processed += 1

      except (OSError, IOError, RuntimeError) as e:
        console.print(f"[red]Failed to process: {e}[/red]")
        repository.update_queue_status(pdf_link.url, "failed", error_message=str(e))
        repository.log_processing(None, "error", f"Failed to process {pdf_link.url}: {e}")
        failed += 1

        if verbose:
          import traceback
          console.print(traceback.format_exc(), style="red")

    # Summary
    console.print("\n[bold blue]Scraping Summary:[/bold blue]")
    console.print(f"  • Total PDFs found: {len(pdf_links)}")
    console.print(f"  • Processed: {processed}")
    console.print(f"  • Skipped (cached): {skipped}")
    console.print(f"  • Failed: {failed}")
    console.print(f"  • Database: {db_path}")
    console.print(f"  • Downloads: {output_dir}")

  except (PDFProcessingError, RuntimeError) as e:
    console.print(f"[red]Error: {e}[/red]")
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
    "• convert - Convert PDFs to CSV using OCR and AI (OpenAI)\n\n"
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