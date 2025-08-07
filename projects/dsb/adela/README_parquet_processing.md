# Chess Game Parquet Processing

This script loads Parquet files containing chess game data, filters them by Elo rating, and parses the movetext column to extract individual chess moves in Standard Algebraic Notation (SAN).

## Features

- **Load multiple Parquet files**: Automatically finds and combines all `.parquet` files in a folder
- **Elo filtering**: Filter games where either player has Elo ≥ threshold (default: 1500)
- **Movetext parsing**: Extract individual chess moves from PGN movetext format
- **Efficient processing**: Uses Polars for fast data processing [[memory:4934569]]
- **Batch processing**: Handle large datasets (40GB+) by processing in memory-efficient batches
- **Memory optimization**: Automatic garbage collection and chunked processing for large files
- **Progress tracking**: Real-time progress bars for long-running operations
- **Command-line interface**: Easy to use with various options for both small and large datasets

## Usage

### Basic Usage (Small Datasets < 5GB)

```bash
uv run python load_and_parse_parquets.py /path/to/parquet/folder
```

### Batch Processing (Large Datasets 5GB+)

**For your 40GB dataset with 1GB files:**
```bash
uv run python load_and_parse_parquets.py data/games/ \
    --batch-mode \
    --batch-size 50000 \
    --output-dir processed_data/ \
    --output-prefix chess_games \
    --min-elo 1500
```

This will:
- Process files one by one to manage memory
- Save results every 50,000 games to `processed_data/chess_games_batch_XXXXXX.parquet`
- Reset the batch and continue processing
- Handle the full 40GB dataset efficiently

### Advanced Options

```bash
# Standard processing with custom Elo threshold
uv run python load_and_parse_parquets.py /path/to/folder --min-elo 1800

# Single file output (for smaller datasets)
uv run python load_and_parse_parquets.py /path/to/folder --output processed_games.parquet

# Sample data for testing (single file mode only)
uv run python load_and_parse_parquets.py /path/to/folder --sample 1000

# Show example parsed moves (single file mode only)
uv run python load_and_parse_parquets.py /path/to/folder --show-examples

# Custom batch size for memory optimization
uv run python load_and_parse_parquets.py /path/to/folder --batch-mode --batch-size 25000

# Custom output directory and file prefix
uv run python load_and_parse_parquets.py /path/to/folder \
    --batch-mode \
    --output-dir /path/to/output/ \
    --output-prefix my_processed_games
```

### Processing Mode Comparison

| Dataset Size | Recommended Mode | Command |
|-------------|------------------|---------|
| < 1GB | Single file mode | `python script.py folder/` |
| 1-5GB | Single file mode with warning | `python script.py folder/ --output result.parquet` |
| 5-40GB+ | Batch mode | `python script.py folder/ --batch-mode` |

### Example for Your 40GB Dataset

```bash
# Process your large dataset efficiently
uv run python load_and_parse_parquets.py data/games/ \
    --batch-mode \
    --min-elo 1500 \
    --batch-size 50000 \
    --output-dir ./processed_high_elo_games/ \
    --output-prefix games_elo1500
```

This will create files like:
- `games_elo1500_batch_000000.parquet` (50,000 games)
- `games_elo1500_batch_000001.parquet` (50,000 games)
- `games_elo1500_batch_000002.parquet` (remaining games)
- ...

## Input Data Format

The script expects Parquet files with the following columns:
- **Elo columns**: `WhiteElo` and `BlackElo` (integer)
- **Movetext column**: `movetext` or `moves` - PGN movetext string containing the game moves
- **Other standard columns**: `White`, `Black`, `Result`, etc.

## Output Format

The processed data includes all original columns plus:
- `parsed_moves`: List of individual chess moves in SAN notation
- `num_moves`: Number of moves in the game

### Batch Processing Output

When using `--batch-mode`, the script creates multiple Parquet files:
- Each batch file contains exactly `--batch-size` games (default: 50,000)
- Final batch may contain fewer games
- Files are numbered sequentially: `prefix_batch_000000.parquet`, `prefix_batch_000001.parquet`, etc.
- Each file is independently readable and contains the same schema

### Memory Efficiency

For large datasets, the batch processing mode:
- Processes files one at a time to limit memory usage
- Uses lazy evaluation with Polars `scan_parquet()` for efficient filtering
- Processes data in 10,000-game chunks within each file
- Performs garbage collection after each file and batch save
- Maintains memory usage under 2-3GB even for 40GB+ datasets

## Move Parsing

The movetext parser handles:
- **Standard moves**: e4, Nf3, Bd5
- **Captures**: exd5, Nxf7
- **Castling**: O-O, O-O-O
- **Promotions**: e8=Q, a1=R+
- **Check/Checkmate**: Qg5+, Qe8#
- **Game results**: Filters out 1-0, 0-1, 1/2-1/2, *

### Example Movetext Parsing

**Input:**
```
1. e4 e6 2. d4 b6 3. a3 Bb7 4. Nc3 Nh6 5. Bxh6 gxh6 6. Be2 Qg5 7. Bg4 h5 8. Nf3 Qg6 9. Nh4 Qg5 10. Bxh5 Qxh4 11. Qf3 Kd8 12. Qxf7 Nc6 13. Qe8# 1-0
```

**Output:**
```
['e4', 'e6', 'd4', 'b6', 'a3', 'Bb7', 'Nc3', 'Nh6', 'Bxh6', 'gxh6', 'Be2', 'Qg5', 'Bg4', 'h5', 'Nf3', 'Qg6', 'Nh4', 'Qg5', 'Bxh5', 'Qxh4', 'Qf3', 'Kd8', 'Qxf7', 'Nc6', 'Qe8#']
```

## Testing

### Basic Functionality Test

Run the test script to see the functionality in action:

```bash
uv run python test_parquet_processing.py
```

This creates sample data and demonstrates the complete pipeline.

### Large Dataset Batch Processing Test

Test the batch processing with simulated large dataset:

```bash
uv run python test_batch_processing.py
```

This creates 75,000 test games and processes them in batches to verify:
- Batch saving at 50,000 game intervals
- Memory efficiency with large datasets
- Proper Elo filtering across batches
- Progress tracking and error handling

## Requirements

- Python 3.9+
- Polars for DataFrame operations [[memory:4934569]]
- uv for package management [[memory:4934577]]

## Project Context

This script is part of the Adela chess analysis project and is designed to work with the existing data processing pipeline. It integrates with the existing PGN parsing infrastructure but focuses specifically on Parquet file processing and movetext extraction.
