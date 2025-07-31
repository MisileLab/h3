#!/usr/bin/env python3
"""
PDF to CSV Converter using Surya OCR and OpenAI Post-processing

This tool extracts text from PDFs using Surya OCR and then uses OpenAI to fix
OCR errors and structure the data for CSV output.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import shutil

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
import pandas as pd
from PIL import Image

from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.table_rec import TableRecPredictor
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

app = typer.Typer(help="Convert PDF to CSV using Surya OCR and OpenAI post-processing")
console = Console()

class PDFToCSVConverter:
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        """Initialize the converter with OpenAI API key."""
        self.openai_api_key = openai_api_key
        self.model = model
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0.1
        )
        
        # Initialize Surya predictors
        console.print("🔄 Initializing Surya OCR models...", style="yellow")
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()
        self.layout_predictor = LayoutPredictor()
        self.table_rec_predictor = TableRecPredictor()
        console.print("✅ Surya OCR models initialized", style="green")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using Surya OCR."""
        console.print(f"📄 Processing PDF: {pdf_path}", style="blue")
        
        # Convert PDF to images
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path)
        except Exception as e:
            console.print(f"❌ Error converting PDF to images: {e}", style="red")
            raise
        
        all_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Extracting text from pages...", total=len(images))
            
            for i, image in enumerate(images):
                progress.update(task, description=f"Processing page {i+1}/{len(images)}")
                
                # Get OCR results
                predictions = self.recognition_predictor(
                    [image], 
                    det_predictor=self.detection_predictor
                )
                
                # Handle the OCR result properly
                if predictions and len(predictions) > 0:
                    prediction = predictions[0]
                    # Check if prediction has text_lines attribute or is a dict
                    if hasattr(prediction, 'text_lines'):
                        text_lines = prediction.text_lines
                    elif isinstance(prediction, dict) and 'text_lines' in prediction:
                        text_lines = prediction['text_lines']
                    else:
                        # Try to convert to dict if it's an OCRResult object
                        try:
                            text_lines = prediction.text_lines if hasattr(prediction, 'text_lines') else []
                        except:
                            text_lines = []
                else:
                    text_lines = []
                
                page_result = {
                    'page': i + 1,
                    'text_lines': text_lines
                }
                all_results.append(page_result)
                
                progress.advance(task)
        
        return all_results
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF using Surya table recognition."""
        console.print(f"📊 Extracting tables from PDF: {pdf_path}", style="blue")
        
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path)
        except Exception as e:
            console.print(f"❌ Error converting PDF to images: {e}", style="red")
            raise
        
        all_tables = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Extracting tables from pages...", total=len(images))
            
            for i, image in enumerate(images):
                progress.update(task, description=f"Processing page {i+1}/{len(images)}")
                
                # Get table recognition results
                table_predictions = self.table_rec_predictor([image])
                
                # Handle table predictions properly
                if table_predictions and len(table_predictions) > 0:
                    page_tables = table_predictions[0]
                    if isinstance(page_tables, list):
                        for table_idx, table_data in enumerate(page_tables):
                            if isinstance(table_data, dict):
                                table_data = table_data.copy()  # Create a copy to avoid modifying original
                                table_data['page'] = i + 1
                                table_data['table_idx'] = table_idx
                                all_tables.append(table_data)
                            else:
                                # Handle non-dict table data
                                table_dict = {
                                    'page': i + 1,
                                    'table_idx': table_idx,
                                    'data': str(table_data)
                                }
                                all_tables.append(table_dict)
                
                progress.advance(task)
        
        return all_tables
    
    def post_process_with_openai(self, text_data: List[Dict[str, Any]], 
                                table_data: List[Dict[str, Any]]) -> str:
        """Post-process OCR results using OpenAI to fix errors and structure data."""
        console.print("🤖 Post-processing with OpenAI...", style="yellow")
        
        # Prepare data for OpenAI
        text_content = self._prepare_text_content(text_data)
        table_content = self._prepare_table_content(table_data)
        
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
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Clean up any markdown formatting that might slip through
            csv_content = response.content.strip()
            
            # Remove markdown code block markers
            if csv_content.startswith('```csv'):
                csv_content = csv_content[7:]  # Remove ```csv
            elif csv_content.startswith('```'):
                csv_content = csv_content[3:]   # Remove ```
            
            if csv_content.endswith('```'):
                csv_content = csv_content[:-3]  # Remove trailing ```
            
            return csv_content.strip()
            
        except Exception as e:
            console.print(f"❌ Error in OpenAI post-processing: {e}", style="red")
            # Fallback to basic CSV conversion
            return self._fallback_csv_conversion(text_data, table_data)
    
    def _prepare_text_content(self, text_data: List[Dict[str, Any]]) -> str:
        """Prepare text content for OpenAI processing."""
        content = []
        for page_data in text_data:
            content.append(f"Page {page_data['page']}:")
            for line in page_data.get('text_lines', []):
                content.append(f"  {safe_get_text(line)}")
            content.append("")
        return "\n".join(content)
    
    def _prepare_table_content(self, table_data: List[Dict[str, Any]]) -> str:
        """Prepare table content for OpenAI processing."""
        if not table_data:
            return "No tables detected"
        
        content = []
        for table in table_data:
            content.append(f"Table on page {table.get('page', 'unknown')}:")
            
            # Extract cell data
            cells = table.get('cells', [])
            if cells:
                # Group cells by row
                rows = {}
                for cell in cells:
                    row_id = cell.get('row_id', 0)
                    col_id = cell.get('col_id', 0)
                    text = cell.get('text', '')
                    if row_id not in rows:
                        rows[row_id] = {}
                    rows[row_id][col_id] = text
                
                # Convert to text representation
                for row_id in sorted(rows.keys()):
                    row_content = []
                    for col_id in sorted(rows[row_id].keys()):
                        row_content.append(rows[row_id][col_id])
                    content.append("  | ".join(row_content))
            
            content.append("")
        
        return "\n".join(content)
    
    def _fallback_csv_conversion(self, text_data: List[Dict[str, Any]], 
                                table_data: List[Dict[str, Any]]) -> str:
        """Fallback CSV conversion when OpenAI fails."""
        console.print("⚠️  Using fallback CSV conversion", style="yellow")
        
        # Simple CSV conversion
        rows = []
        headers = ['Page', 'Line', 'Text', 'Confidence']
        
        for page_data in text_data:
            page_num = page_data['page']
            for i, line in enumerate(page_data.get('text_lines', [])):
                rows.append([
                    page_num,
                    i + 1,
                    safe_get_text(line),
                    safe_get_confidence(line)
                ])
        
        # Create CSV
        df = pd.DataFrame(rows, columns=headers)
        return df.to_csv(index=False)
    
    def save_csv(self, csv_content: str, output_path: str):
        """Save CSV content to file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)
            console.print(f"✅ CSV saved to: {output_path}", style="green")
        except Exception as e:
            console.print(f"❌ Error saving CSV: {e}", style="red")
            raise

def validate_openai_key(api_key: str) -> bool:
    """Validate OpenAI API key format."""
    return api_key.startswith('sk-') and len(api_key) > 20

def safe_get_text(line_obj) -> str:
    """Safely extract text from a TextLine object or dict."""
    if hasattr(line_obj, 'text'):
        return line_obj.text
    elif isinstance(line_obj, dict) and 'text' in line_obj:
        return line_obj['text']
    else:
        return str(line_obj)

def safe_get_confidence(line_obj) -> float:
    """Safely extract confidence from a TextLine object or dict."""
    if hasattr(line_obj, 'confidence'):
        return line_obj.confidence
    elif isinstance(line_obj, dict) and 'confidence' in line_obj:
        return line_obj['confidence']
    else:
        return 0.0

def _show_preview(text_data: List[Dict[str, Any]], table_data: List[Dict[str, Any]]):
    """Show preview of extracted data."""
    # Show text preview
    if text_data:
        console.print("\n📝 Text Preview:", style="yellow")
        for page_data in text_data[:2]:  # Show first 2 pages
            console.print(f"Page {page_data['page']}:")
            for i, line in enumerate(page_data.get('text_lines', [])[:5]):  # Show first 5 lines
                text = safe_get_text(line)[:100]  # Truncate long text
                console.print(f"  {i+1}: {text}")
            console.print("")
    
    # Show table preview
    if table_data:
        console.print("📊 Table Preview:", style="yellow")
        for table in table_data[:2]:  # Show first 2 tables
            console.print(f"Table on page {table.get('page', 'unknown')}:")
            cells = table.get('cells', [])
            if cells:
                # Show first few cells
                for cell in cells[:10]:
                    text = cell.get('text', '')[:50]
                    console.print(f"  Cell ({cell.get('row_id', 0)}, {cell.get('col_id', 0)}): {text}")
            console.print("")

@app.command()
def convert(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file"),
    output_path: str = typer.Option(None, "--output", "-o", help="Output CSV file path"),
    openai_api_key: str = typer.Option(None, "--api-key", help="OpenAI API key"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="OpenAI model to use"),
    extract_tables: bool = typer.Option(True, "--tables/--no-tables", help="Extract tables from PDF"),
    show_preview: bool = typer.Option(False, "--preview", help="Show preview of extracted data"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Convert PDF to CSV using Surya OCR and OpenAI post-processing."""
    
    # Validate inputs
    if not os.path.exists(pdf_path):
        console.print(f"❌ PDF file not found: {pdf_path}", style="red")
        raise typer.Exit(1)
    
    # Get OpenAI API key
    if not openai_api_key:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            console.print("❌ OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key", style="red")
            raise typer.Exit(1)
    
    if not validate_openai_key(openai_api_key):
        console.print("❌ Invalid OpenAI API key format", style="red")
        raise typer.Exit(1)
    
    # Set output path
    if not output_path:
        pdf_name = Path(pdf_path).stem
        output_path = f"{pdf_name}_output.csv"
    
    try:
        # Initialize converter
        converter = PDFToCSVConverter(openai_api_key, model)
        
        # Extract text
        text_data = converter.extract_text_from_pdf(pdf_path)
        
        # Extract tables if requested
        table_data = []
        if extract_tables:
            table_data = converter.extract_tables_from_pdf(pdf_path)
        
        # Show preview if requested
        if show_preview:
            console.print("\n📋 Preview of extracted data:", style="blue")
            _show_preview(text_data, table_data)
        
        # Post-process with OpenAI
        csv_content = converter.post_process_with_openai(text_data, table_data)
        
        # Save CSV
        converter.save_csv(csv_content, output_path)
        
        # Show summary
        console.print("\n📊 Conversion Summary:", style="blue")
        console.print(f"  • Pages processed: {len(text_data)}")
        console.print(f"  • Text lines extracted: {sum(len(page.get('text_lines', [])) for page in text_data)}")
        console.print(f"  • Tables detected: {len(table_data)}")
        console.print(f"  • Output file: {output_path}")
        
    except Exception as e:
        console.print(f"❌ Error during conversion: {e}", style="red")
        if verbose:
            import traceback
            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1)

@app.command()
def info():
    """Show information about the tool and dependencies."""
    console.print(Panel.fit(
        "[bold blue]PDF to CSV Converter[/bold blue]\n\n"
        "This tool uses Surya OCR for text extraction and OpenAI for post-processing.\n\n"
        "[bold]Features:[/bold]\n"
        "• Multi-language OCR support\n"
        "• Table detection and extraction\n"
        "• AI-powered OCR error correction\n"
        "• Layout analysis\n"
        "• Reading order detection\n\n"
        "[bold]Dependencies:[/bold]\n"
        "• Surya OCR for text extraction\n"
        "• OpenAI API for post-processing\n"
        "• PDF2Image for PDF conversion\n"
        "• Pandas for CSV handling",
        title="Tool Information"
    ))

if __name__ == "__main__":
    app() 