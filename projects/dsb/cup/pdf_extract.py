#!/usr/bin/env python3
"""
PDF Text Extractor - Extract text from PDFs without OCR

This tool extracts text directly from PDFs that contain text layers,
without using OCR. It's much faster and more accurate than OCR for
PDFs that already have text content.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli import app

if __name__ == "__main__":
  app() 