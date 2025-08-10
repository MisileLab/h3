"""Self-play-only training entrypoint.

This script intentionally supports only self-play for training.
"""

import argparse
from typing import Optional

from examples.train_model import train_from_self_play


def main() -> None:
  parser = argparse.ArgumentParser(description="Train Adela via self-play only")
  parser.add_argument("--output-dir", type=str, default="models/self_play", help="Directory to save checkpoints")
  parser.add_argument("--games", type=int, default=100, help="Number of self-play games to generate")
  parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
  parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
  parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
  parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
  parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum improvement in val_loss to reset patience")
  parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda or cpu")

  args = parser.parse_args()

  # Train strictly from self-play
  train_from_self_play(
    output_dir=args.output_dir,
    num_games=args.games,
    num_epochs=args.epochs,
    batch_size=args.batch_size,
    mcts_simulations=args.simulations,
    early_stop_patience=args.patience,
    early_stop_min_delta=args.min_delta,
    device=args.device if isinstance(args.device, str) else None,
  )


if __name__ == "__main__":
  main()
