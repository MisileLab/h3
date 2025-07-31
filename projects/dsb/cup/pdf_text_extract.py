#!/usr/bin/env python3
"""
PDF Text Extractor - Extract text from PDFs without OCR

This tool extracts text directly from PDFs that contain text layers,
without using OCR. It's much faster and more accurate than OCR for
PDFs that already have text content.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
import pandas as pd

app = typer.Typer(help="Extract text from PDFs without OCR")
console = Console()

class PDFTextExtractor:
    def __init__(self):
        """Initialize the text extractor."""
        console.print("📄 Initializing PDF text extractor...", style="blue")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using direct text extraction."""
        console.print(f"📄 Processing PDF: {pdf_path}", style="blue")
        
        try:
            import pypdf
        except ImportError:
            console.print("❌ pypdf is required. Install with: pip install pypdf", style="red")
            raise
        
        all_results = []
        
        try:
            with open(pdf_path, 'rb') as file:
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
                        
                        # Split text into lines and clean up
                        lines = []
                        if text:
                            for line_num, line in enumerate(text.split('\n'), 1):
                                line = line.strip()
                                if line:  # Only include non-empty lines
                                    lines.append({
                                        'line_number': line_num,
                                        'text': line,
                                        'page': page_num + 1
                                    })
                        
                        page_result = {
                            'page': page_num + 1,
                            'text_lines': lines,
                            'total_lines': len(lines),
                            'raw_text': text
                        }
                        all_results.append(page_result)
                        
                        progress.advance(task)
                
                console.print(f"✅ Extracted text from {total_pages} pages", style="green")
                
        except Exception as e:
            console.print(f"❌ Error reading PDF: {e}", style="red")
            raise
        
        return all_results
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF using tabula-py."""
        console.print(f"📊 Extracting tables from PDF: {pdf_path}", style="blue")
        
        try:
            import tabula
        except ImportError:
            console.print("⚠️  tabula-py not available. Install with: pip install tabula-py", style="yellow")
            console.print("   Skipping table extraction.", style="yellow")
            return []
        
        all_tables = []
        
        try:
            # Extract all tables from the PDF
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Processing tables...", total=len(tables))
                
                for table_idx, table in enumerate(tables):
                    progress.update(task, description=f"Processing table {table_idx + 1}/{len(tables)}")
                    
                    # Convert table to list of dictionaries
                    table_data = []
                    for row_idx, row in table.iterrows():
                        row_dict = {}
                        for col_idx, value in enumerate(row):
                            col_name = table.columns[col_idx] if col_idx < len(table.columns) else f"Column_{col_idx}"
                            row_dict[col_name] = str(value) if pd.notna(value) else ""
                        table_data.append(row_dict)
                    
                    table_result = {
                        'table_index': table_idx + 1,
                        'rows': table_data,
                        'columns': list(table.columns),
                        'shape': table.shape
                    }
                    all_tables.append(table_result)
                    
                    progress.advance(task)
                
                console.print(f"✅ Extracted {len(tables)} tables", style="green")
                
        except Exception as e:
            console.print(f"❌ Error extracting tables: {e}", style="red")
            console.print("   Continuing without table extraction.", style="yellow")
        
        return all_tables
    
    def save_text_to_file(self, text_data: List[Dict[str, Any]], output_path: str):
        """Save extracted text to a text file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for page_data in text_data:
                    f.write(f"\n=== Page {page_data['page']} ===\n\n")
                    for line in page_data['text_lines']:
                        f.write(f"{line['text']}\n")
                    f.write("\n")
            
            console.print(f"✅ Text saved to: {output_path}", style="green")
        except Exception as e:
            console.print(f"❌ Error saving text file: {e}", style="red")
            raise
    
    def save_to_csv(self, text_data: List[Dict[str, Any]], output_path: str):
        """Save extracted text to CSV format."""
        try:
            rows = []
            headers = ['Page', 'Line', 'Text']
            
            for page_data in text_data:
                page_num = page_data['page']
                for line in page_data['text_lines']:
                    rows.append([
                        page_num,
                        line['line_number'],
                        line['text']
                    ])
            
            df = pd.DataFrame(rows, columns=headers)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            console.print(f"✅ CSV saved to: {output_path}", style="green")
        except Exception as e:
            console.print(f"❌ Error saving CSV: {e}", style="red")
            raise
    
    def save_tables_to_csv(self, table_data: List[Dict[str, Any]], output_path: str):
        """Save extracted tables to CSV format."""
        if not table_data:
            console.print("⚠️  No tables to save", style="yellow")
            return
        
        try:
            # For single table, save in normal table format
            if len(table_data) == 1:
                table = table_data[0]
                df = pd.DataFrame(table['rows'])
                df.to_csv(output_path, index=False, encoding='utf-8')
                console.print(f"✅ Table CSV saved to: {output_path}", style="green")
            else:
                # For multiple tables, save each to separate files
                base_path = output_path.replace('.csv', '')
                for table in table_data:
                    table_idx = table['table_index']
                    table_path = f"{base_path}_table_{table_idx}.csv"
                    df = pd.DataFrame(table['rows'])
                    df.to_csv(table_path, index=False, encoding='utf-8')
                    console.print(f"✅ Table {table_idx} CSV saved to: {table_path}", style="green")
                
                # Also create a combined file with table identification
                combined_rows = []
                for table in table_data:
                    table_idx = table['table_index']
                    for row in table['rows']:
                        row_with_table = {'Table': f"Table_{table_idx}", **row}
                        combined_rows.append(row_with_table)
                
                combined_df = pd.DataFrame(combined_rows)
                combined_path = f"{base_path}_combined.csv"
                combined_df.to_csv(combined_path, index=False, encoding='utf-8')
                console.print(f"✅ Combined tables CSV saved to: {combined_path}", style="green")
                
        except Exception as e:
            console.print(f"❌ Error saving tables CSV: {e}", style="red")
            raise
    
    def save_to_json(self, text_data: List[Dict[str, Any]], table_data: List[Dict[str, Any]], output_path: str):
        """Save extracted data to JSON format."""
        try:
            result = {
                'pdf_path': str(Path(output_path).parent),
                'extraction_method': 'direct_text',
                'pages': text_data,
                'tables': table_data,
                'summary': {
                    'total_pages': len(text_data),
                    'total_text_lines': sum(len(page['text_lines']) for page in text_data),
                    'total_tables': len(table_data)
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            console.print(f"✅ JSON saved to: {output_path}", style="green")
        except Exception as e:
            console.print(f"❌ Error saving JSON: {e}", style="red")
            raise

def _show_preview(text_data: List[Dict[str, Any]], table_data: List[Dict[str, Any]]):
    """Show preview of extracted data."""
    # Show text preview
    if text_data:
        console.print("\n📝 Text Preview:", style="yellow")
        for page_data in text_data[:2]:  # Show first 2 pages
            console.print(f"Page {page_data['page']} ({page_data['total_lines']} lines):")
            for line in page_data['text_lines'][:5]:  # Show first 5 lines
                text = line['text'][:100]  # Truncate long text
                console.print(f"  {line['line_number']}: {text}")
            console.print("")
    
    # Show table preview
    if table_data:
        console.print("📊 Table Preview:", style="yellow")
        for table in table_data[:2]:  # Show first 2 tables
            console.print(f"Table {table['table_index']} ({table['shape'][0]} rows, {table['shape'][1]} columns):")
            if table['rows']:
                # Show first few rows
                for row_idx, row in enumerate(table['rows'][:3], 1):
                    row_text = " | ".join(f"{k}: {v}" for k, v in list(row.items())[:3])
                    console.print(f"  Row {row_idx}: {row_text}")
            console.print("")

@app.command()
def extract(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file"),
    output_format: str = typer.Option("csv", "--format", "-f", help="Output format: csv, txt, json, or all"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    extract_tables: bool = typer.Option(True, "--tables/--no-tables", help="Extract tables from PDF"),
    show_preview: bool = typer.Option(False, "--preview", help="Show preview of extracted data"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Extract text from PDF without OCR."""
    
    # Validate inputs
    if not os.path.exists(pdf_path):
        console.print(f"❌ PDF file not found: {pdf_path}", style="red")
        raise typer.Exit(1)
    
    # Set output directory
    if not output_dir:
        pdf_name = Path(pdf_path).stem
        output_dir = f"{pdf_name}_text_extract"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize extractor
        extractor = PDFTextExtractor()
        
        # Extract text
        text_data = extractor.extract_text_from_pdf(pdf_path)
        
        # Extract tables if requested
        table_data = []
        if extract_tables:
            table_data = extractor.extract_tables_from_pdf(pdf_path)
        
        # Show preview if requested
        if show_preview:
            console.print("\n📋 Preview of extracted data:", style="blue")
            _show_preview(text_data, table_data)
        
        # Save in requested format(s)
        pdf_name = Path(pdf_path).stem
        
        if output_format.lower() in ['csv', 'all']:
            csv_path = os.path.join(output_dir, f"{pdf_name}_text.csv")
            extractor.save_to_csv(text_data, csv_path)
        
        if output_format.lower() in ['txt', 'all']:
            txt_path = os.path.join(output_dir, f"{pdf_name}_text.txt")
            extractor.save_text_to_file(text_data, txt_path)
        
        if output_format.lower() in ['json', 'all']:
            json_path = os.path.join(output_dir, f"{pdf_name}_extract.json")
            extractor.save_to_json(text_data, table_data, json_path)
        
        if extract_tables and table_data and output_format.lower() in ['csv', 'all']:
            tables_csv_path = os.path.join(output_dir, f"{pdf_name}_tables.csv")
            extractor.save_tables_to_csv(table_data, tables_csv_path)
            
            # Also save a properly formatted table CSV
            if len(table_data) == 1:
                # For single table, create a clean CSV with proper headers
                table = table_data[0]
                clean_csv_path = os.path.join(output_dir, f"{pdf_name}_table_clean.csv")
                df = pd.DataFrame(table['rows'])
                df.to_csv(clean_csv_path, index=False, encoding='utf-8')
                console.print(f"✅ Clean table CSV saved to: {clean_csv_path}", style="green")
        
        # Show summary
        console.print("\n📊 Extraction Summary:", style="blue")
        console.print(f"  • Pages processed: {len(text_data)}")
        console.print(f"  • Text lines extracted: {sum(len(page['text_lines']) for page in text_data)}")
        console.print(f"  • Tables detected: {len(table_data)}")
        console.print(f"  • Output directory: {output_dir}")
        
    except Exception as e:
        console.print(f"❌ Error during extraction: {e}", style="red")
        if verbose:
            import traceback
            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1)

@app.command()
def info():
    """Show information about the tool."""
    console.print(Panel.fit(
        "[bold blue]PDF Text Extractor[/bold blue]\n\n"
        "This tool extracts text directly from PDFs without using OCR.\n\n"
        "[bold]Features:[/bold]\n"
        "• Direct text extraction (no OCR required)\n"
        "• Table extraction using tabula-py\n"
        "• Multiple output formats (CSV, TXT, JSON)\n"
        "• Fast processing\n"
        "• High accuracy for text-based PDFs\n\n"
        "[bold]When to use:[/bold]\n"
        "• PDFs with text layers (most modern PDFs)\n"
        "• Scanned documents converted to text\n"
        "• Forms and documents with embedded text\n\n"
        "[bold]When NOT to use:[/bold]\n"
        "• Pure image-based PDFs\n"
        "• Scanned documents without OCR\n"
        "• Handwritten documents",
        title="Tool Information"
    ))

if __name__ == "__main__":
    app() 