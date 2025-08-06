# Adela Examples

This directory contains example scripts for using the Adela chess engine.

## Running Examples

You can run these examples using the `uv run` command:

```bash
# Run the simple game example
uv run python examples/simple_game.py

# Run the training example
uv run python examples/train_model.py
```

Alternatively, you can use the provided `run_example.py` script:

```bash
# Run the simple game example
uv run python run_example.py simple_game

# Run the training example
uv run python run_example.py train_model
```

## Available Examples

1. **Simple Game** (`simple_game.py`)
   - Demonstrates how to use the engine to play a simple game
   - Shows expert contributions for each move
   - Includes an interactive mode to play against the engine

2. **Train Model** (`train_model.py`)
   - Shows how to train the model using PGN data or self-play
   - Demonstrates the training pipeline
   - Includes validation and model saving
