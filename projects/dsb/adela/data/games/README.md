# Local Training Data Structure

This directory contains the local training data for the Adela chess AI. The training pipeline expects data to be organized in three splits:

## Directory Structure

```
data/games/
├── train/          # Training data parquet files
├── validation/     # Validation data parquet files  
├── test/           # Test data parquet files
└── README.md       # This file
```

## File Format

Each subdirectory should contain parquet files with chess game data. The expected format includes:

### Required Columns
- **`parsed_moves`**: List of individual moves (e.g., ["e4", "e5", "Nf3", ...])
- **`num_moves`**: Number of moves in the game (integer)
- **`Result`** or **`result`**: Game result ("1-0", "0-1", "1/2-1/2")

### Optional Columns (for filtering)
- **`WhiteElo`** or **`white_elo`**: White player's Elo rating
- **`BlackElo`** or **`black_elo`**: Black player's Elo rating

### Legacy Columns (no longer used)
- **`moves`** or **`movetext`**: Raw PGN move notation - no longer processed automatically

## Data Preparation

**IMPORTANT**: Your parquet files must already contain `parsed_moves` and `num_moves` columns. Use the preprocessing utilities in this project to convert raw PGN data.

### Step 1: Preprocess Raw Data
If you have raw PGN files or parquet files without `parsed_moves`, use the preprocessing tools:

```python
# Example preprocessing (adjust paths as needed)
from load_and_parse_parquets import add_parsed_moves
import polars as pl

# Load raw parquet file
df = pl.read_parquet("raw_games.parquet")

# Add parsed_moves and num_moves columns
df = add_parsed_moves(df, chunk_size=1000)

# Save processed file
df.write_parquet("processed_games.parquet")
```

### Step 2: Split Into Train/Validation/Test

#### Option A: Manual Split
1. Place your **pre-processed** parquet files in the appropriate subdirectories
2. Ensure roughly 80% in `train/`, 10% in `validation/`, 10% in `test/`

#### Option B: Use the Split Utility
If you have a single directory of **pre-processed** parquet files:

```python
from adela.data.hf_dataset import split_and_upload_parquet

# Split files into train/validation/test locally
split_and_upload_parquet(
    source_folder="path/to/your/processed/parquet/files",
    repo_id="your-local-dataset",  # Can be any name
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

## Training Usage

Once your data is organized, train the model using:

```python
from examples.train_model import train_from_local_data

train_from_local_data(
    data_path="data/games",  # This directory
    output_dir="models/adela",
    num_epochs=50,
    batch_size=256,
    min_elo=1000,  # Minimum Elo for game inclusion
    early_stop_patience=5,
    early_stop_min_delta=1e-3,
)
```

Or simply run:
```bash
uv run train_model.py
```

## Data Sources

Recommended sources for chess game data:
- [Lichess Database](https://database.lichess.org/) - Monthly game exports
- [Chess.com Game Archives](https://www.chess.com/games/archive) - Member games
- [FICS Game Database](http://www.ficsgames.org/) - Free Internet Chess Server games
- [TWIC Archives](https://theweekinchess.com/twic) - The Week in Chess

Convert PGN files to parquet using the included utilities in this project.

## Notes

- Ensure parquet files contain games with sufficient Elo ratings (recommended: ≥1000)
- The training pipeline will filter games based on the `min_elo` parameter
- Games with unparseable moves will be skipped automatically
- Monitor memory usage when processing large datasets
