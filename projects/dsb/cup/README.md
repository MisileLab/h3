# PDF Processing Tools

A comprehensive collection of tools for extracting and converting PDF content to various formats with **database storage** and **automatic web scraping**. This project provides both direct text extraction (for PDFs with text layers) and OCR-based extraction (for scanned documents) with optional AI-powered post-processing using OpenAI.

## ✨ Features

- **Direct Text Extraction**: Extract text from PDFs with text layers (no OCR required)
- **OCR-based Extraction**: Use Surya OCR for scanned documents and images
- **Table Detection**: Automatically detect and extract tables from PDFs
- **AI Post-processing**: Use OpenAI to fix OCR errors and structure data
- **Multiple Output Formats**: CSV, TXT, JSON, and combined formats
- **Address and Coordinate Lookup**: Automatically find nearest address and coordinates using Kakao Local API
- **Database Storage**: Store all extraction results in SQLite database
- **Web Scraping**: Automatically scrape and process PDFs from government websites
- **Smart Caching**: Skip already processed files to avoid duplicates
- **Secrets Management**: Integrated with Infisical for secure API key management
- **Automated Workflows**: Just command runner for simplified operations
- **Retry Logic**: Automatic retry for failed network requests with exponential backoff
- **Fast Processing**: Optimized for speed and accuracy
- **Rich CLI Interface**: Beautiful command-line interface with progress bars

## 🚀 Quick Start

### Prerequisites

- Python 3.13.5+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- [just](https://github.com/casey/just) command runner (optional, for automation)
- [Infisical CLI](https://infisical.com/docs/cli/overview) (optional, for secrets management)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd cup
   ```

2. **Install dependencies**:
   ```bash
   python install.py
   # Or manually
   uv sync --extra dev
   # Or with just
   just install
   ```

3. **Set up environment variables**:

   **Option A: Using Infisical (Recommended for production)**
   ```bash
   # Install Infisical CLI
   # See: https://infisical.com/docs/cli/overview

   # Login to Infisical
   infisical login

   # Set secrets in Infisical dashboard or CLI
   infisical secrets set OPENAI_API_KEY=sk-your-api-key-here
   infisical secrets set KAKAO_API_KEY=your-kakao-api-key

   # Verify secrets
   just check-env
   ```

   **Option B: Using .env file (For local development)**
   ```bash
   # Create .env file from template
   cat > .env << 'EOF'
   OPENAI_API_KEY=sk-your-api-key-here
   KAKAO_API_KEY=your-kakao-api-key
   EOF
   ```

   **Option C: Export environment variables**
   ```bash
   # Required for AI post-processing
   export OPENAI_API_KEY='sk-your-api-key-here'

   # Required for address lookup
   export KAKAO_API_KEY='your-kakao-api-key'
   ```

## 📖 Usage

### 1. Basic Text Extraction

Extract text from a PDF with text layers:

```bash
# Extract to CSV (default)
uv run python pdf_extract.py extract document.pdf

# Extract to multiple formats
uv run python pdf_extract.py extract document.pdf --format all

# Extract with preview
uv run python pdf_extract.py extract document.pdf --preview

# Extract with reference coordinates for accurate address lookup
uv run python pdf_extract.py extract document.pdf --ref-x 957123.45 --ref-y 1943789.56
```

### 2. Extract with Database Storage

Store extraction results in SQLite database:

```bash
# Extract and store in database
uv run python pdf_extract.py extract document.pdf --store-db

# Specify custom database path
uv run python pdf_extract.py extract document.pdf --store-db --db-path ./my_data.db

# Extract with OCR and store in database
uv run python pdf_extract.py extract scanned_doc.pdf --ocr --store-db
```

### 3. OCR-based Extraction

Extract text from scanned documents using OCR:

```bash
# Use OCR for text extraction
uv run python pdf_extract.py extract scanned_document.pdf --ocr

# Extract tables with OCR
uv run python pdf_extract.py extract scanned_document.pdf --ocr --tables
```

### 4. Web Scraping (NEW!)

Automatically scrape PDFs from government websites and process them:

```bash
# Scrape all PDFs from Dongjak government portal (default URL)
uv run python pdf_extract.py scrape

# Scrape with PDF limit
uv run python pdf_extract.py scrape --limit 10

# Scrape only one page (first page of bulletin board)
uv run python pdf_extract.py scrape --max-pages 1

# Scrape first 3 pages with max 50 PDFs
uv run python pdf_extract.py scrape --max-pages 3 --limit 50

# Scrape with OCR
uv run python pdf_extract.py scrape --ocr

# Scrape with custom database and output directory
uv run python pdf_extract.py scrape --db-path data.db --output-dir ./pdfs

# Scrape with reference coordinates for better address matching
uv run python pdf_extract.py scrape --ref-x 957123.45 --ref-y 1943789.56

# Process all without skipping existing files
uv run python pdf_extract.py scrape --no-skip-existing

# Scrape from custom URL
uv run python pdf_extract.py scrape https://example.com/board
```

**What happens during scraping:**
1. 🔍 Scrapes all PDF links from the bulletin board
2. 📥 Downloads PDFs to local directory
3. 📝 Extracts text and tables from each PDF
4. 🗺️ Looks up addresses using Kakao Local API
5. 💾 Stores everything in SQLite database
6. 🚫 Skips already processed files (smart caching)

### 5. Using Just Commands (Automated Workflows)

The `justfile` provides convenient commands for common workflows:

```bash
# Show all available commands
just

# Flexible scraping with page and PDF limits
just scrape "URL"              # Scrape all pages, all PDFs
just scrape "URL" 1            # Scrape first page only, all PDFs
just scrape "URL" 1 50         # Scrape first page, max 50 PDFs
just scrape "URL" 5 100        # Scrape first 5 pages, max 100 PDFs

# Convenience aliases
just scrape-all "URL"          # Scrape all pages, all PDFs
just scrape-test "URL"         # Scrape all pages, first 10 PDFs
just scrape-test "URL" 20      # Scrape all pages, first 20 PDFs

# Extract from single PDF
just extract document.pdf
just extract-ocr scanned.pdf

# Extract with LLM post-processing
just extract-llm document.pdf
just extract-llm-ocr scanned.pdf

# Database operations
just db-info      # Show database statistics
just db-queue     # Show pending scraping queue items

# Development commands
just test         # Run tests
just test-cov     # Run tests with coverage
just format       # Format code with black
just lint         # Lint with flake8
just quality      # Run all quality checks

# Full workflow: scrape all from Dongjak portal
just workflow-dongjak "https://www.dongjak.go.kr/portal/bbs/B0000591/list.do?menuNo=200209"
```

**Benefits of using `just`:**
- ✅ Automatically uses Infisical for secrets management
- ✅ Shorter, more memorable commands
- ✅ Flexible page and PDF limits with simple syntax
- ✅ Consistent command interface
- ✅ Built-in documentation (`just --list`)

### 6. AI-powered Conversion

Convert PDFs to CSV using OCR and AI post-processing:

```bash
# Convert with AI post-processing (default: gpt-4o-mini)
uv run python pdf_convert.py convert document.pdf

# Specify output file
uv run python pdf_convert.py convert document.pdf --output result.csv

# Use different OpenAI model
uv run python pdf_convert.py convert document.pdf --model gpt-4o

# Convert with direct extraction + LLM
uv run python pdf_extract.py extract-llm document.pdf

# Convert with OCR + LLM
uv run python pdf_extract.py extract-llm document.pdf --ocr
```

## 🎯 Command Reference

### `extract` - Extract text and tables

```bash
uv run python pdf_extract.py extract [OPTIONS] PDF_PATH
```

**Options:**
- `--format, -f TEXT`: Output format (csv, txt, json, all) [default: csv]
- `--output-dir, -o TEXT`: Output directory
- `--tables/--no-tables`: Extract tables [default: tables]
- `--preview`: Show preview of extracted data
- `--ocr`: Use OCR for text extraction
- `--ref-x FLOAT`: Reference X coordinate (WTM format)
- `--ref-y FLOAT`: Reference Y coordinate (WTM format)
- `--store-db`: Store results in database
- `--db-path TEXT`: SQLite database path [default: ./data.db]
- `--verbose, -v`: Verbose output

### `scrape` - Scrape and process PDFs (NEW!)

```bash
uv run python pdf_extract.py scrape [OPTIONS] [URL]
```

**Arguments:**
- `URL`: URL of bulletin board [default: Dongjak government portal]

**Options:**
- `--limit, -l INTEGER`: Limit number of PDFs [default: 0 = no limit]
- `--max-pages INTEGER`: Maximum number of pages to scrape [default: None = unlimited]
- `--ocr`: Use OCR for extraction
- `--skip-existing`: Skip already downloaded files [default: true]
- `--db-path TEXT`: SQLite database path [default: ./data.db]
- `--output-dir, -o TEXT`: Download directory [default: ./downloads/dongjak]
- `--ref-x FLOAT`: Reference X coordinate (WTM format)
- `--ref-y FLOAT`: Reference Y coordinate (WTM format)
- `--verbose, -v`: Verbose output

### `convert` - AI-powered conversion

```bash
uv run python pdf_convert.py convert [OPTIONS] PDF_PATH
```

**Options:**
- `--output, -o TEXT`: Output CSV file path
- `--api-key TEXT`: OpenAI API key
- `--model, -m TEXT`: OpenAI model [default: gpt-4o-mini]
- `--tables/--no-tables`: Extract tables [default: tables]
- `--preview`: Show preview of extracted data
- `--ref-x FLOAT`: Reference X coordinate (WTM format)
- `--ref-y FLOAT`: Reference Y coordinate (WTM format)
- `--verbose, -v`: Verbose output

### `extract-llm` - Extract + LLM post-processing

```bash
uv run python pdf_extract.py extract-llm [OPTIONS] PDF_PATH
```

**Options:**
- `--output, -o TEXT`: Output CSV file path
- `--api-key TEXT`: OpenAI API key
- `--model, -m TEXT`: OpenAI model [default: gpt-4o-mini]
- `--ocr`: Use OCR for text extraction
- `--ref-x FLOAT`: Reference X coordinate (WTM format)
- `--ref-y FLOAT`: Reference Y coordinate (WTM format)
- `--verbose, -v`: Verbose output

### `info` - Show tool information

```bash
uv run python pdf_extract.py info
```

## 🗄️ Database Schema

The SQLite database contains 7 tables:

### `documents`
Stores PDF metadata and processing information:
- `pdf_path`, `original_filename`, `source_url`
- `total_pages`, `total_text_lines`, `total_tables`
- `extraction_method`, `processing_time`, `status`

### `pages`
Page-level data:
- `document_id`, `page_number`, `total_lines`, `raw_text`

### `text_lines`
Individual text lines with location data:
- `page_id`, `line_number`, `text`, `confidence`
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`
- `nearest_address`, `coordinate_x`, `coordinate_y` (from Kakao API)

### `tables`
Table metadata:
- `page_id`, `table_index`, `num_rows`, `num_columns`
- `columns_json`, `address_column`

### `table_rows`
Table row data with enriched location information:
- `table_id`, `row_index`, `row_data_json`
- `place_name`, `approval_amount`
- `nearest_address`, `coordinate_x`, `coordinate_y` (from Kakao API)

### `scraping_queue`
Tracks scraped URLs and processing status:
- `url`, `title`, `status`, `download_path`
- `added_date`, `processed_date`, `error_message`

### `processing_logs`
Stores processing logs for debugging:
- `document_id`, `log_level`, `message`, `timestamp`

## 📁 Project Structure

```
cup/
├── src/                          # Source code
│   ├── core/                     # Core functionality
│   │   ├── config.py            # Configuration management
│   │   ├── exceptions.py        # Custom exceptions
│   │   ├── types.py             # Type definitions
│   │   └── kakao_api.py         # Kakao Local API integration
│   ├── extractors/              # PDF extraction modules
│   │   ├── base.py              # Base extractor class
│   │   ├── direct_text.py       # Direct text extraction
│   │   └── ocr_text.py          # OCR-based extraction
│   ├── output/                  # Output formatting
│   │   ├── base.py              # Base formatter class
│   │   ├── csv_formatter.py     # CSV output with Kakao API
│   │   ├── text_formatter.py    # Text output
│   │   └── json_formatter.py    # JSON output
│   ├── processors/              # AI post-processing
│   │   ├── base.py              # Base processor class
│   │   └── llm_processor.py     # OpenAI integration
│   ├── db/                      # Database module (NEW!)
│   │   ├── __init__.py
│   │   ├── schema.py            # Database schema
│   │   └── repository.py        # CRUD operations
│   ├── scrapers/                # Web scraping (NEW!)
│   │   ├── __init__.py
│   │   ├── base.py              # Base scraper class
│   │   └── dongjak_scraper.py   # Dongjak portal scraper
│   ├── app.py                   # Main application class
│   └── cli.py                   # Command-line interface
├── pdf_extract.py               # Text extraction entry point
├── pdf_convert.py               # AI conversion entry point
├── install.py                   # Installation script
├── pyproject.toml              # Project configuration
├── CLAUDE.md                   # Development guide
└── README.md                   # This file
```

## 🏗️ Architecture

### Core Layer (`src/core/`)
- **Configuration**: Centralized settings including API keys and model defaults
- **Type Definitions**: Pydantic models for type-safe data structures
- **Kakao API**: Integration with Kakao Local API for address/coordinate lookup
- **Exceptions**: Custom exceptions for error handling

### Extraction Layer (`src/extractors/`)
- **Base Extractor**: Abstract interface
- **Direct Text**: Uses pypdf for text-based PDFs
- **OCR Text**: Uses Surya OCR for scanned documents

### Output Layer (`src/output/`)
- **CSV Formatter**: With async Kakao API integration for address enrichment
- **Text Formatter**: Plain text output
- **JSON Formatter**: Structured JSON output

### Processing Layer (`src/processors/`)
- **LLM Processor**: OpenAI integration for AI post-processing

### Database Layer (`src/db/`) - NEW!
- **Schema**: SQLite schema definitions
- **Repository**: CRUD operations and caching logic

### Scraping Layer (`src/scrapers/`) - NEW!
- **Base Scraper**: Abstract interface for scrapers
- **Dongjak Scraper**: Scraper for Dongjak government portal
  - Uses requests + BeautifulSoup (fast)
  - Selenium fallback available (if needed)

### Application Layer
- **PDFProcessor**: Orchestrates extraction, formatting, and database storage
- **CLI**: Rich command-line interface using Typer

## 🔧 Configuration

### Environment Variables

**Required:**
- `OPENAI_API_KEY`: OpenAI API key for AI post-processing
- `KAKAO_API_KEY`: Kakao API key for address and coordinate lookup

**Optional (Performance Tuning):**
- `RECOGNITION_BATCH_SIZE`: OCR recognition batch size (default: 32 CPU, 512 GPU)
- `DETECTOR_BATCH_SIZE`: OCR detector batch size (default: 6 CPU, 36 GPU)
- `TABLE_REC_BATCH_SIZE`: Table recognition batch size (default: 8 CPU, 64 GPU)
- `TORCH_DEVICE`: Device for OCR processing (default: cpu, options: cpu, cuda)

### Performance Tuning

**For GPU acceleration:**
```bash
export TORCH_DEVICE=cuda
export RECOGNITION_BATCH_SIZE=512
export DETECTOR_BATCH_SIZE=36
export TABLE_REC_BATCH_SIZE=64
```

**For CPU optimization:**
```bash
export TORCH_DEVICE=cpu
export RECOGNITION_BATCH_SIZE=32
export DETECTOR_BATCH_SIZE=6
export TABLE_REC_BATCH_SIZE=8
```

## 🎓 Use Cases

### Use `extract` when:
- ✅ PDFs have text layers (most modern PDFs)
- ✅ You need fast extraction without OCR
- ✅ Working with forms or documents with embedded text
- ✅ You want multiple output formats

### Use `extract --ocr` when:
- ✅ Working with scanned documents
- ✅ PDFs are image-based
- ✅ Documents have poor text layer quality

### Use `extract --store-db` when:
- ✅ You need to track processing history
- ✅ You want to query extracted data with SQL
- ✅ You need to avoid reprocessing files
- ✅ You want to store Kakao API address results

### Use `scrape` when:
- ✅ You need to process many PDFs from a website automatically
- ✅ You want automatic download + extraction + database storage
- ✅ You need to keep results updated from a bulletin board
- ✅ You want smart caching to avoid duplicate processing

### Use `convert` or `extract-llm` when:
- ✅ You need AI-powered error correction
- ✅ You want structured CSV output
- ✅ Working with low-quality OCR results
- ✅ You need intelligent data structuring

## 💾 Database Queries

After processing, you can query the database:

```bash
# Open database
sqlite3 data.db

# View all documents
SELECT pdf_path, status, total_pages FROM documents;

# View processed URLs
SELECT url, status, processed_date FROM scraping_queue WHERE status = 'processed';

# View table rows with addresses
SELECT place_name, approval_amount, nearest_address, coordinate_x, coordinate_y
FROM table_rows
WHERE nearest_address IS NOT NULL;

# View processing errors
SELECT url, error_message FROM scraping_queue WHERE status = 'failed';
```

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --extra dev

# Install pre-commit hooks (if available)
pre-commit install
```

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **flake8**: Linting

Run all quality checks:
```bash
black src/
isort src/
mypy src/
flake8 src/
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_core.py
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
- Make sure you're using Python 3.13.5+
- Install dependencies: `python install.py`

**OCR Performance Issues**
- Adjust batch sizes based on your hardware
- Use GPU if available: `export TORCH_DEVICE=cuda`

**OpenAI API Errors**
- Verify your API key: `echo $OPENAI_API_KEY`
- Check your usage limits on OpenAI dashboard
- Ensure key starts with `sk-`

**Kakao API Errors**
- Verify your API key: `echo $KAKAO_API_KEY`
- Check Kakao API quota and permissions

**Database Errors**
- Check file permissions on database path
- Ensure database directory exists
- Verify SQLite3 is installed: `sqlite3 --version`

**Web Scraping Issues**
- Check network connectivity
- Verify the URL is accessible
- Use `--verbose` flag for detailed error messages
- Try with `--limit 1` to test single file first

### Getting Help

```bash
# Show tool information
uv run python pdf_extract.py info

# Get help for specific command
uv run python pdf_extract.py extract --help
uv run python pdf_extract.py scrape --help

# Use verbose mode for debugging
uv run python pdf_extract.py extract file.pdf --verbose
```

## 📝 Examples

### Example 1: Process Korean transaction PDFs with address lookup

```bash
# Set reference coordinates (example for Seoul)
export REF_X=957123.45
export REF_Y=1943789.56

# Process single PDF with address lookup
uv run python pdf_extract.py extract transactions.pdf \
  --ref-x $REF_X \
  --ref-y $REF_Y \
  --store-db \
  --db-path transactions.db
```

### Example 2: Scrape and process government announcements

```bash
# Scrape first 20 PDFs with OCR
uv run python pdf_extract.py scrape \
  --limit 20 \
  --ocr \
  --db-path government.db \
  --output-dir ./pdfs/government

# Or using just: scrape first 3 pages with max 50 PDFs
just scrape "https://example.com/board" 3 50
```

### Example 3: Batch processing with AI post-processing

```bash
# Process all PDFs in directory with OpenAI
for pdf in pdfs/*.pdf; do
  uv run python pdf_extract.py extract-llm "$pdf" \
    --store-db \
    --db-path processed.db
done
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all quality checks pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Surya OCR](https://github.com/VikParuchuri/surya) - OCR capabilities
- [OpenAI](https://openai.com/) - AI post-processing
- [Kakao Developers](https://developers.kakao.com/) - Local API for address lookup
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Beautiful terminal output
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) - Web scraping

## 🚀 What's New

### Version 2.1 (Latest)

**New Features:**
- 🔐 **Infisical Integration**: Secure secrets management with Infisical CLI
- ⚡ **Just Commands**: Automated workflows with `justfile` for simplified operations
- 🔄 **Retry Logic**: Automatic retry with exponential backoff for network requests
- 🎯 **Enhanced Scraper**: Configurable pagination limits and request delays
- 📝 **Dotenv Support**: Local development fallback with `.env` files

**Improvements:**
- Better error handling with detailed error messages
- Improved scraping reliability with 3-attempt retry
- More flexible configuration options
- Enhanced documentation with justfile examples

### Version 2.0

**Major Features:**
- ✨ **Database Storage**: Store all extraction results in SQLite
- 🌐 **Web Scraping**: Automatically scrape PDFs from government websites
- 🗺️ **Kakao API Integration**: Automatic address and coordinate lookup
- 🚫 **Smart Caching**: Avoid reprocessing already downloaded files
- 🔄 **OpenAI Integration**: Replaced OpenRouter with OpenAI API

**New Commands:**
- `scrape` - Automatically download and process PDFs from websites
- `--store-db` - Store results in database
- `--db-path` - Specify custom database location

**Database Schema:**
- 7 tables with full relational structure
- Stores text, tables, addresses, coordinates
- Processing logs and scraping queue tracking
