# PDF to CSV Converter

A powerful CLI tool that converts PDF files to CSV using **Surya OCR** for text extraction and **OpenAI** for post-processing to fix OCR errors and structure data.

## Features

- 🔍 **Multi-language OCR support** - Works with 90+ languages
- 📊 **Table detection and extraction** - Automatically identifies and extracts tables
- 🤖 **AI-powered error correction** - Uses OpenAI to fix OCR errors
- 📐 **Layout analysis** - Understands document structure
- 📖 **Reading order detection** - Maintains logical document flow
- 🎯 **High accuracy** - Benchmarks favorably vs cloud services
- ⚡ **Fast processing** - Optimized for both CPU and GPU

## Installation

### Prerequisites

- Python 3.10+
- OpenAI API key
- (Optional) GPU for faster processing

### Install Dependencies

```bash
uv sync
```

The project already includes Surya OCR and other required dependencies in `pyproject.toml`.

### Using Environment Variables

1. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. **Convert a PDF to CSV:**
   ```bash
   python pdf_to_csv.py convert your_document.pdf
   ```

3. **View the results:**
   The tool will create `your_document_output.csv` with the extracted and processed data.

## Usage

```bash
# Convert PDF to CSV
python pdf_to_csv.py convert document.pdf

# Specify output file
python pdf_to_csv.py convert document.pdf --output result.csv

# Show preview of extracted data
python pdf_to_csv.py convert document.pdf --preview
```

### Advanced Options

```bash
# Use a different OpenAI model
python pdf_to_csv.py convert document.pdf --model gpt-4o

# Extract only text (no tables)
python pdf_to_csv.py convert document.pdf --no-tables

# Verbose output for debugging
python pdf_to_csv.py convert document.pdf --verbose

# Provide API key directly
python pdf_to_csv.py convert document.pdf --api-key sk-your-key-here
```

### Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output, -o` | Output CSV file path | `{filename}_output.csv` |
| `--api-key` | OpenAI API key | `OPENAI_API_KEY` env var |
| `--model, -m` | OpenAI model to use | `gpt-4o-mini` |
| `--tables/--no-tables` | Extract tables from PDF | `True` |
| `--preview` | Show preview of extracted data | `False` |
| `--verbose, -v` | Verbose output | `False` |

### Available Commands

```bash
# Convert PDF to CSV
python pdf_to_csv.py convert <pdf_file>

# Show tool information
python pdf_to_csv.py info

# Show help
python pdf_to_csv.py --help
```

## How It Works

1. **PDF Processing**: Converts PDF pages to images using `pdf2image`
2. **Text Extraction**: Uses Surya OCR to detect and extract text lines
3. **Table Detection**: Identifies and extracts table structures
4. **AI Post-processing**: Sends extracted data to OpenAI for:
   - OCR error correction (fixing 0/O, 1/l, 5/S, etc.)
   - Data structuring and formatting
   - CSV generation with appropriate headers
5. **Output**: Saves clean, structured CSV file

## Example Output

### Input PDF
A document with tables, text, and potential OCR errors.

### Output CSV
```csv
Name,Age,Department,Salary
John Doe,32,Engineering,75000
Jane Smith,28,Marketing,65000
Bob Johnson,35,Sales,80000
```

## Error Handling

The tool includes robust error handling:

- **Fallback CSV conversion** if OpenAI processing fails
- **Graceful degradation** for missing dependencies
- **Detailed error messages** with verbose mode
- **Input validation** for PDF files and API keys

## Performance Tips

### GPU Acceleration
Set environment variables for optimal performance:

```bash
# For GPU processing
export TORCH_DEVICE=cuda
export RECOGNITION_BATCH_SIZE=512
export DETECTOR_BATCH_SIZE=36
export TABLE_REC_BATCH_SIZE=64
```

### CPU Optimization
```bash
# For CPU processing
export RECOGNITION_BATCH_SIZE=32
export DETECTOR_BATCH_SIZE=6
export TABLE_REC_BATCH_SIZE=8
```

## Troubleshooting

### Common Issues

1. **"PDF file not found"**
   - Ensure the PDF file path is correct
   - Check file permissions

2. **"OpenAI API key not provided"**
   - Set `OPENAI_API_KEY` environment variable
   - Or use `--api-key` option

3. **"Invalid OpenAI API key format"**
   - API key should start with `sk-`
   - Check for typos or extra spaces

4. **Slow processing**
   - Use GPU if available
   - Adjust batch sizes for your hardware
   - Consider using `gpt-4o-mini` instead of `gpt-4o`

5. **OCR quality issues**
   - Ensure PDF has good resolution
   - Try preprocessing (binarizing, deskewing)
   - Adjust Surya thresholds if needed

### Debug Mode

Use verbose mode for detailed error information:

```bash
python pdf_to_csv.py convert document.pdf --verbose
```

## API Reference

### PDFToCSVConverter Class

```python
from pdf_to_csv import PDFToCSVConverter

# Initialize
converter = PDFToCSVConverter(openai_api_key, model="gpt-4o-mini")

# Extract text
text_data = converter.extract_text_from_pdf("document.pdf")

# Extract tables
table_data = converter.extract_tables_from_pdf("document.pdf")

# Post-process with OpenAI
csv_content = converter.post_process_with_openai(text_data, table_data)

# Save CSV
converter.save_csv(csv_content, "output.csv")
```

## Examples

See `example_usage.py` for complete usage examples:

```bash
python example_usage.py
```

## Dependencies

- **Surya OCR**: Multi-language document OCR
- **OpenAI**: AI-powered post-processing
- **PDF2Image**: PDF to image conversion
- **Pandas**: CSV handling
- **Rich**: Beautiful terminal output
- **Typer**: CLI framework

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

- Check the [troubleshooting section](#troubleshooting)
- Open an issue for bugs or feature requests

## Acknowledgments

- [Surya OCR](https://github.com/datalab-to/surya) for the excellent OCR capabilities
- [OpenAI](https://openai.com/) for the AI post-processing
- The open source community for the supporting libraries
