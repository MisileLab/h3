# Adela

A modern Mixture-of-Experts chess AI using PyTorch and MCTS.

## Features

- Multiple expert networks specialized by game phase, play style, and opponent adaptation
- Dynamic gating system to select and weight experts based on current board state
- Integration of human gameplay patterns from Lichess data
- Combined MCTS and neural network evaluation for efficient search
- Self-play and human-data hybrid training pipeline

## Installation

```bash
# Clone the repository
git clone https://gith.misile.xyz:/projects/dsb/adela.git
cd adela

# Install the package
uv sync
```

## Project Structure

- `adela/core/`: Chess board representation and utilities
- `adela/experts/`: Expert network implementations
- `adela/gating/`: Dynamic expert gating system
- `adela/mcts/`: Monte Carlo Tree Search implementation
- `adela/opponent/`: Opponent analysis and adaptation
- `adela/training/`: Training pipeline and data processing
- `adela/evaluation/`: Evaluation metrics and tools
- `adela/data/`: Lichess data pipeline for downloading and processing chess data

## Usage

### Command Line Interface

```bash
# Play a game against the engine
uv run python run.py play

# Analyze a position
uv run python run.py analyze --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Engine plays against itself
uv run python run.py selfplay --games 5

# Run the Lichess data pipeline
uv run python run_lichess_pipeline.py --games --puzzles --evaluations --num-months 1 --min-elo 2000
```

### Python API

```python
from adela.engine import AdelaEngine

# Initialize the engine
engine = AdelaEngine()

# Get the best move for a position
best_move = engine.get_best_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

# Play a game against the engine
engine.play_game()

# Get expert contributions for current position
contributions = engine.get_expert_contributions()
```

### Lichess Data Pipeline

```python
from adela.data.pipeline import LichessDataPipeline

# Initialize the pipeline
pipeline = LichessDataPipeline(data_dir="./data")

# Download and process chess games
games_path = pipeline.process_games(
    num_months=1,
    min_elo=2000,
    max_games=10000
)

# Download and process chess puzzles
puzzles_path = pipeline.process_puzzles(
    min_rating=1500,
    max_rating=2500,
    themes_filter=["mate", "endgame"],
    max_puzzles=10000
)

# Download and process chess position evaluations
evals_path = pipeline.process_evaluations(
    min_depth=20,
    max_evaluations=10000
)

# Load and analyze the processed data
import polars as pl
games_df = pl.read_parquet(games_path)
puzzles_df = pl.read_parquet(puzzles_path)
evals_df = pl.read_parquet(evals_path)
```

## Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
uv run python run_tests.py
# Or directly with pytest
uv run pytest -xvs tests

# Format code
uv run black adela tests
uv run isort adela tests
```

## License

MIT
