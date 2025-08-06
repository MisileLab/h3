# Processing Large PGN Files (200GB+)

This document explains how to efficiently process very large PGN files (200GB+) using the specialized tools in this project.

## Overview

The standard PGN parser in the Lichess data pipeline works well for moderately sized files, but for extremely large files (200GB+), we need a more memory-efficient approach that processes the file line by line.

## Tools Provided

1. **LargePGNParser**: A specialized parser that processes PGN files line by line, saving batches to Parquet format.
2. **process_large_pgn.py**: A command-line script to process large PGN files.
3. **analyze_chess_data.py**: A script to analyze the processed Parquet files.
4. **examples/process_sample_pgn.py**: An example script demonstrating how to use the parser.

## Usage

### Processing a Large PGN File

```bash
# Process a large PGN file with default settings
uv run python process_large_pgn.py path/to/large_file.pgn

# Process with custom settings
uv run python process_large_pgn.py path/to/large_file.pgn \
  --output-dir ./data/processed \
  --batch-size 100000 \
  --min-elo 1800 \
  --output-prefix lichess_games_2025 \
  --num-workers 8 \
  --no-merge
```

### Command-line Arguments

- `pgn_file`: Path to the PGN file to process
- `--output-dir`: Directory to save Parquet files (default: "./data/games")
- `--batch-size`: Number of games to process in a batch (default: 50000)
- `--min-elo`: Minimum Elo rating for games to include (default: 0)
- `--output-prefix`: Prefix for output Parquet filenames (default: "chess_games_YYYYMMDD")
- `--no-merge`: Don't merge batch files into a single Parquet file
- `--no-progress`: Don't show progress bar
- `--num-workers`: Number of worker threads to use for parallel processing (default: 0 = use all CPU cores)

### Analyzing the Processed Data

```bash
# Analyze the processed data
uv run python analyze_chess_data.py path/to/processed_games.parquet

# Analyze with custom settings
uv run python analyze_chess_data.py path/to/processed_games.parquet \
  --output-dir ./analysis \
  --sample-size 10000 \
  --min-elo 2000 \
  --no-plots
```

### Command-line Arguments

- `parquet_file`: Path to the Parquet file to analyze
- `--output-dir`: Directory to save analysis results (default: "./analysis")
- `--sample-size`: Number of games to sample for analysis (default: all games)
- `--min-elo`: Minimum Elo rating for games to include (default: 0)
- `--no-plots`: Don't generate plots

## Example

```bash
# Run the example script
uv run python examples/process_sample_pgn.py
```

## Data Format

The processed data is stored in Parquet format with the following columns:

- `id`: Game ID
- `event`: Event name
- `site`: Site URL
- `date`: Game date
- `time`: Game time
- `white`: White player name
- `black`: Black player name
- `white_elo`: White player Elo rating
- `black_elo`: Black player Elo rating
- `white_rating_diff`: White player rating change
- `black_rating_diff`: Black player rating change
- `result`: Game result ("1-0", "0-1", "1/2-1/2", "*")
- `time_control`: Time control (e.g., "60+0")
- `base_time_seconds`: Base time in seconds
- `increment_seconds`: Increment in seconds
- `termination`: Termination reason
- `eco`: ECO code
- `opening`: Opening name
- `moves`: Space-separated list of all moves in sequential order
- `num_moves`: Total number of moves
- `has_clock_times`: Whether the game has clock times
- `avg_time_per_move`: Average time per move (seconds)
- `min_time_per_move`: Minimum time per move (seconds)
- `max_time_per_move`: Maximum time per move (seconds)

## Performance Considerations

- The parser processes the file line by line, so memory usage remains constant regardless of file size.
- Data is saved in batches to avoid memory issues.
- The batch size can be adjusted based on available memory.
- Multi-threading support is available to speed up processing on multi-core systems using a producer-consumer pattern:
  - One reader thread reads the PGN file line by line and puts games into a queue
  - Multiple worker threads process games from the queue in parallel
  - One saver thread saves processed games in batches
- By default, the parser uses all available CPU cores for maximum performance.
- Processing a 200GB PGN file will take several hours, depending on hardware.
- Consider running the process in the background or on a dedicated server.

## Troubleshooting

- **Memory Issues**: Reduce the batch size if you encounter memory problems.
- **Parsing Errors**: The parser is designed to be robust and will skip malformed games.
- **Performance**: Increase the batch size for faster processing if you have sufficient memory.

## Advanced Usage

### Python API

You can also use the parser directly in your Python code:

```python
from adela.data.large_pgn_parser import LargePGNParser

# Initialize parser
parser = LargePGNParser(
    batch_size=50000,
    output_dir="./data/processed",
    min_elo=1800,
    output_filename_prefix="lichess_games_2025",
    num_workers=8  # Use 8 worker threads for parallel processing
)

# Parse PGN file
games_processed = parser.parse_pgn_file(
    pgn_path="path/to/large_file.pgn",
    show_progress=True
)

# Merge batch files
merged_path = parser.merge_parquet_files()

# Analyze the data
import polars as pl
df = pl.read_parquet(merged_path)
print(f"Number of games: {len(df)}")
```
