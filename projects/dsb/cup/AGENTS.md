# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PDF processing toolkit that extracts text and tables from PDFs using both direct text extraction and OCR (Surya OCR), with optional AI post-processing via OpenAI API. The primary use case is processing Korean PDFs containing merchant transaction data with automatic address lookup via Kakao Local API.

## Common Commands

### Installation
```bash
# Install dependencies (uses uv package manager)
python install.py

# Or manually with uv
uv sync --extra dev
```

### Running the Tools

**Direct text extraction (for PDFs with text layers):**
```bash
python pdf_extract.py extract document.pdf --format csv
python pdf_extract.py extract document.pdf --format all --preview
python pdf_extract.py extract document.pdf --ref-x 957123.45 --ref-y 1943789.56
```

**OCR-based extraction (for scanned documents):**
```bash
python pdf_extract.py extract document.pdf --ocr
python pdf_extract.py extract document.pdf --ocr --tables
```

**AI-powered conversion (OCR + LLM post-processing):**
```bash
python pdf_convert.py convert document.pdf
python pdf_convert.py convert document.pdf --model gpt-4o
```

**Direct/OCR + LLM post-processing:**
```bash
python pdf_extract.py extract_llm document.pdf
python pdf_extract.py extract_llm document.pdf --ocr
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_core.py
```

### Code Quality
```bash
black src/
isort src/
mypy src/
flake8 src/
```

## Architecture

### Modular Design with Clear Separation

The codebase follows a clean architecture with distinct layers:

**Core Layer** (`src/core/`)
- `config.py`: Centralized configuration including batch sizes, API keys, model defaults
- `types.py`: Pydantic models for type-safe data structures (TextLine, TableData, PageData, ProcessingResult, CSVRow)
- `exceptions.py`: Custom exceptions for error handling
- `kakao_api.py`: Integration with Kakao Local API for address/coordinate lookup

**Extraction Layer** (`src/extractors/`)
- `base.py`: Abstract BaseExtractor interface
- `direct_text.py`: Uses pypdf for text-based PDFs and tabula-py for tables
- `ocr_text.py`: Uses Surya OCR models (recognition, detection, layout, table recognition)

**Output Layer** (`src/output/`)
- `base.py`: Abstract BaseOutputFormatter interface
- `csv_formatter.py`: CSV output with async Kakao API integration for address enrichment
- `text_formatter.py`: Plain text output
- `json_formatter.py`: Structured JSON output

**Processing Layer** (`src/processors/`)
- `base.py`: Abstract BaseProcessor interface
- `llm_processor.py`: OpenAI integration for AI post-processing (error correction, data structuring)

**Application Layer**
- `app.py`: PDFProcessor orchestrates extraction, formatting, and processing
- `cli.py`: Typer-based CLI with commands: extract, convert, extract_llm, info

### Key Architectural Patterns

**Extractor Pattern**: Both extractors return `list[PageData]`, making them interchangeable. The app selects the appropriate extractor based on `use_ocr` flag.

**Formatter Pattern**: All formatters implement `format()` and `save()` methods. CSV formatter has special async logic to enrich data with Kakao API address lookups.

**Async Address Enrichment**: The CSV formatter performs async batch lookups against Kakao API:
1. Extracts place names from table columns (esp. "가맹점명")
2. Searches Kakao API with optional WTM reference coordinates
3. Finds nearest matching place using WTM coordinate distance calculation
4. Adds address, x, y coordinates to output rows

**OpenAI Integration**: LLM processor uses httpx to call OpenAI API for post-processing extracted text, fixing OCR errors and structuring data.

### Data Flow

1. **Input**: PDF file path
2. **Extraction**: DirectTextExtractor or OCRTextExtractor → `list[PageData]`
3. **Combination**: Text pages + Table pages merged by page number
4. **Result Creation**: `ProcessingResult` aggregates all pages with metadata
5. **Formatting**: Selected formatter(s) convert to output format
6. **Address Enrichment** (CSV only): Async Kakao API lookups add location data
7. **Output**: Files saved to `{pdf_name}_extract/` directory

### Important Implementation Details

**OCR Models Initialization**: OCRTextExtractor initializes Surya's foundation predictor and shares it across recognition, detection, layout, and table recognition predictors. This is memory-intensive and should be done once.

**Batch Size Configuration**: OCR performance is tuned via environment variables:
- `RECOGNITION_BATCH_SIZE`: Default 32 (CPU), 512 (GPU)
- `DETECTOR_BATCH_SIZE`: Default 6 (CPU), 36 (GPU)
- `TABLE_REC_BATCH_SIZE`: Default 8 (CPU), 64 (GPU)
- `TORCH_DEVICE`: "cpu" or "cuda"

**Reference Coordinates**: WTM (Western Transverse Mercator) coordinates can be provided via `--ref-x` and `--ref-y` to find the nearest address match when multiple results exist.

**Place Name Mapping**: `csv_formatter.py` contains a `place_name_mapping` dict to translate English/alternative place names to Korean for Kakao API queries (e.g., "wagamama" → "와가마마").

**Table Column Detection**: CSV formatter intelligently searches for place name columns using keywords: "가맹점", "상호", "매장", "장소", "place", "store", "shop". Falls back to heuristics if not found.

**Amount Conversion**: CSV formatter converts the "승인금액" (approval amount) column from formatted strings to integers by removing commas and quotes.

## Environment Variables

Required for full functionality:
- `OPENAI_API_KEY`: For AI post-processing (default model: "gpt-4o-mini")
- `KAKAO_API_KEY`: For address and coordinate lookup

Optional performance tuning:
- `RECOGNITION_BATCH_SIZE`, `DETECTOR_BATCH_SIZE`, `TABLE_REC_BATCH_SIZE`
- `TORCH_DEVICE`: "cpu" or "cuda"

## Dependencies

Key libraries and their purposes:
- **pypdf**: Direct text extraction from PDFs
- **tabula-py**: Table extraction for non-OCR PDFs
- **surya-ocr**: OCR engine (text recognition, detection, layout, table recognition)
- **pdf2image**: Convert PDF pages to images for OCR
- **polars**: Fast DataFrame operations for CSV output
- **pydantic**: Type-safe data models with validation
- **typer**: CLI framework
- **rich**: Beautiful console output with progress bars
- **httpx**: Async HTTP client for API calls
- **langchain/langchain-openai**: (Present but not actively used in main flow)

## Development Notes

- Python 3.13.5 required (see pyproject.toml)
- Uses `uv` package manager (not pip)
- Entry points: `pdf_extract.py` and `pdf_convert.py` both import from `src.cli`
- Legacy scripts `pdf_text_extract.py` and `pdf_to_csv.py` exist but are not the main entry points
- Output directory pattern: `{pdf_name}_extract/` containing `{pdf_name}_{format}.{format}` files
- Table outputs: `{pdf_name}_table.csv` for single table, `{pdf_name}_table_1.csv`, `{pdf_name}_table_2.csv`, etc. for multiple tables, plus `{pdf_name}_combined.csv`
