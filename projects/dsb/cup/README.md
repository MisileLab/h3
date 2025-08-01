# PDF Processing Tools

A comprehensive collection of tools for extracting and converting PDF content to various formats. This project provides both direct text extraction (for PDFs with text layers) and OCR-based extraction (for scanned documents) with optional AI-powered post-processing.

## Features

- **Direct Text Extraction**: Extract text from PDFs with text layers (no OCR required)
- **OCR-based Extraction**: Use Surya OCR for scanned documents and images
- **Table Detection**: Automatically detect and extract tables from PDFs
- **AI Post-processing**: Use OpenAI to fix OCR errors and structure data
- **Multiple Output Formats**: CSV, TXT, JSON, and combined formats
- **Fast Processing**: Optimized for speed and accuracy
- **Rich CLI Interface**: Beautiful command-line interface with progress bars

## Installation

### Prerequisites

- Python 3.13.5
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd cup
   ```

2. **Install dependencies**:
   ```bash
   python install.py
   ```

3. **Set up OpenAI API key** (optional, for AI post-processing):
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

### Basic Text Extraction

Extract text from a PDF with text layers:

```bash
# Extract to CSV (default)
python pdf_extract.py extract document.pdf

# Extract to multiple formats
python pdf_extract.py extract document.pdf --format all

# Extract to specific format
python pdf_extract.py extract document.pdf --format json

# Show preview of extracted data
python pdf_extract.py extract document.pdf --preview
```

### OCR-based Extraction

Extract text from scanned documents using OCR:

```bash
# Use OCR for text extraction
python pdf_extract.py extract scanned_document.pdf --ocr

# Extract tables with OCR
python pdf_extract.py extract scanned_document.pdf --ocr --tables
```

### AI-powered Conversion

Convert PDFs to CSV using OCR and AI post-processing:

```bash
# Convert with AI post-processing
python pdf_convert.py convert document.pdf

# Specify output file
python pdf_convert.py convert document.pdf --output result.csv

# Use different OpenAI model
python pdf_convert.py convert document.pdf --model gpt-4
```

### Command Options

#### Extract Command
- `--format, -f`: Output format (csv, txt, json, all)
- `--output-dir, -o`: Output directory
- `--tables/--no-tables`: Extract tables (default: true)
- `--preview`: Show preview of extracted data
- `--ocr`: Use OCR for text extraction
- `--verbose, -v`: Verbose output

#### Convert Command
- `--output, -o`: Output CSV file path
- `--api-key`: OpenAI API key
- `--model, -m`: OpenAI model to use (default: gpt-4o-mini)
- `--tables/--no-tables`: Extract tables (default: true)
- `--preview`: Show preview of extracted data
- `--verbose, -v`: Verbose output

## Project Structure

```
cup/
├── src/                          # Source code
│   ├── core/                     # Core functionality
│   │   ├── config.py            # Configuration management
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── types.py             # Type definitions
│   ├── extractors/              # PDF extraction modules
│   │   ├── base.py              # Base extractor class
│   │   ├── direct_text.py       # Direct text extraction
│   │   └── ocr_text.py          # OCR-based extraction
│   ├── output/                  # Output formatting
│   │   ├── base.py              # Base formatter class
│   │   ├── csv_formatter.py     # CSV output
│   │   ├── text_formatter.py    # Text output
│   │   └── json_formatter.py    # JSON output
│   ├── processors/              # AI post-processing
│   │   ├── base.py              # Base processor class
│   │   └── openai_processor.py  # OpenAI integration
│   ├── app.py                   # Main application class
│   └── cli.py                   # Command-line interface
├── pdf_extract.py               # Text extraction entry point
├── pdf_convert.py               # AI conversion entry point
├── install.py                   # Installation script
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Architecture

The project follows a modular architecture with clear separation of concerns:

### Core Module (`src/core/`)
- **Configuration Management**: Centralized settings and constants
- **Type Definitions**: Strongly typed data structures
- **Custom Exceptions**: Domain-specific error handling

### Extractors (`src/extractors/`)
- **Base Extractor**: Abstract interface for all extractors
- **Direct Text Extractor**: Uses pypdf for text-based PDFs
- **OCR Text Extractor**: Uses Surya OCR for scanned documents

### Output Formatters (`src/output/`)
- **Base Formatter**: Abstract interface for all formatters
- **CSV Formatter**: Structured CSV output with table support
- **Text Formatter**: Plain text output
- **JSON Formatter**: Structured JSON output

### Processors (`src/processors/`)
- **Base Processor**: Abstract interface for AI processors
- **OpenAI Processor**: AI-powered post-processing and error correction

### Application Layer
- **PDFProcessor**: Main application class orchestrating all components
- **CLI Interface**: Rich command-line interface using Typer

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for AI post-processing
- `RECOGNITION_BATCH_SIZE`: OCR recognition batch size (default: 32)
- `DETECTOR_BATCH_SIZE`: OCR detector batch size (default: 6)
- `TABLE_REC_BATCH_SIZE`: Table recognition batch size (default: 8)
- `TORCH_DEVICE`: Device for OCR processing (default: cpu)

### Performance Tuning

For GPU acceleration:
```bash
export TORCH_DEVICE=cuda
export RECOGNITION_BATCH_SIZE=512
export DETECTOR_BATCH_SIZE=36
export TABLE_REC_BATCH_SIZE=64
```

For CPU optimization:
```bash
export RECOGNITION_BATCH_SIZE=32
export DETECTOR_BATCH_SIZE=6
export TABLE_REC_BATCH_SIZE=8
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --extra dev

# Install pre-commit hooks
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
```

## When to Use Each Tool

### Use `pdf_extract.py` when:
- PDFs have text layers (most modern PDFs)
- You need fast extraction without OCR
- Working with forms or documents with embedded text
- You want multiple output formats

### Use `pdf_convert.py` when:
- Working with scanned documents
- PDFs are image-based
- You need AI-powered error correction
- You want structured CSV output
- Working with handwritten documents

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're using Python 3.10+ and have installed all dependencies
2. **OCR Performance**: Adjust batch sizes based on your hardware
3. **OpenAI API Errors**: Verify your API key and check your usage limits
4. **Memory Issues**: Reduce batch sizes for large documents

### Getting Help

- Check the tool information: `python pdf_extract.py info`
- Use verbose mode for detailed error messages: `--verbose`
- Check the logs for specific error details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all quality checks pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Surya OCR](https://github.com/VikParuchuri/surya) for OCR capabilities
- [OpenAI](https://openai.com/) for AI post-processing
- [Typer](https://typer.tiangolo.com/) for the CLI framework
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output
