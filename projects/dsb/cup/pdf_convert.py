#!/usr/bin/env python3
"""
PDF to CSV Converter using Surya OCR and OpenAI Post-processing

This tool extracts text from PDFs using Surya OCR and then uses OpenAI to fix
OCR errors and structure the data for CSV output.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli import app

if __name__ == "__main__":
  app()