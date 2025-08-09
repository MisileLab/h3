"""Command-line tools for dataset upload/download on Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from .hf_dataset import split_and_upload_parquet


def main() -> int:
  parser = argparse.ArgumentParser(description="Adela HF dataset tools")
  sub = parser.add_subparsers(dest="cmd", required=True)

  up = sub.add_parser("upload", help="Split local Parquet files and upload to HF")
  up.add_argument("source", type=str, help="Folder with .parquet files")
  up.add_argument("repo", type=str, help="HF dataset repo id, e.g. misilelab/adela-dataset")
  up.add_argument("--train", type=float, default=0.8, dest="train_ratio")
  up.add_argument("--val", type=float, default=0.1, dest="val_ratio")
  up.add_argument("--test", type=float, default=0.1, dest="test_ratio")
  up.add_argument("--message", type=str, default="Upload dataset with auto splits")

  args = parser.parse_args()

  if args.cmd == "upload":
    split_and_upload_parquet(
      source_folder=Path(args.source),
      repo_id=args.repo,
      train_ratio=args.train_ratio,
      val_ratio=args.val_ratio,
      test_ratio=args.test_ratio,
      commit_message=args.message,
    )
    print(f"Uploaded splits to {args.repo}")
    return 0

  return 1


if __name__ == "__main__":
  raise SystemExit(main())

"""Command-line interface for the Lichess data pipeline."""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from adela.data.pipeline import LichessDataPipeline

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments.

  Returns:
    Parsed arguments.
  """
  parser = argparse.ArgumentParser(description="Lichess data pipeline")
  
  # General options
  parser.add_argument(
    "--data-dir",
    type=str,
    default="./data",
    help="Directory to store processed data"
  )
  parser.add_argument(
    "--temp-dir",
    type=str,
    default=None,
    help="Directory to store temporary files"
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=50000,
    help="Number of records to process in a batch"
  )
  parser.add_argument(
    "--keep-temp-files",
    action="store_true",
    help="Keep temporary files after processing"
  )
  
  # Data type options
  parser.add_argument(
    "--games",
    action="store_true",
    help="Process chess games"
  )
  parser.add_argument(
    "--puzzles",
    action="store_true",
    help="Process chess puzzles"
  )
  parser.add_argument(
    "--evaluations",
    action="store_true",
    help="Process chess position evaluations"
  )
  
  # Game options
  parser.add_argument(
    "--num-months",
    type=int,
    default=1,
    help="Number of months of games to download"
  )
  parser.add_argument(
    "--min-elo",
    type=int,
    default=1800,
    help="Minimum Elo rating for games to include"
  )
  parser.add_argument(
    "--max-games",
    type=int,
    default=None,
    help="Maximum number of games to process"
  )
  
  # Puzzle options
  parser.add_argument(
    "--min-puzzle-rating",
    type=int,
    default=0,
    help="Minimum puzzle rating to include"
  )
  parser.add_argument(
    "--max-puzzle-rating",
    type=int,
    default=None,
    help="Maximum puzzle rating to include"
  )
  parser.add_argument(
    "--puzzle-themes",
    type=str,
    default=None,
    help="Comma-separated list of puzzle themes to include"
  )
  parser.add_argument(
    "--max-puzzles",
    type=int,
    default=None,
    help="Maximum number of puzzles to process"
  )
  
  # Evaluation options
  parser.add_argument(
    "--min-eval-depth",
    type=int,
    default=20,
    help="Minimum evaluation depth to include"
  )
  parser.add_argument(
    "--max-evaluations",
    type=int,
    default=None,
    help="Maximum number of evaluations to process"
  )
  
  return parser.parse_args()


def main() -> int:
  """Main entry point for the command-line interface.

  Returns:
    Exit code.
  """
  args = parse_args()
  
  # Process puzzle themes
  puzzle_themes = None
  if args.puzzle_themes:
    puzzle_themes = args.puzzle_themes.split(",")
  
  # If no data types are specified, process all
  if not (args.games or args.puzzles or args.evaluations):
    args.games = True
    args.puzzles = True
    args.evaluations = True
  
  # Create pipeline
  pipeline = LichessDataPipeline(
    data_dir=args.data_dir,
    temp_dir=args.temp_dir,
    batch_size=args.batch_size
  )
  
  # Run pipeline
  results = pipeline.run_full_pipeline(
    process_games=args.games,
    process_puzzles=args.puzzles,
    process_evaluations=args.evaluations,
    num_months=args.num_months,
    min_elo=args.min_elo,
    max_games=args.max_games,
    min_puzzle_rating=args.min_puzzle_rating,
    max_puzzle_rating=args.max_puzzle_rating,
    puzzle_themes=puzzle_themes,
    max_puzzles=args.max_puzzles,
    min_eval_depth=args.min_eval_depth,
    max_evaluations=args.max_evaluations,
    delete_temp_files=not args.keep_temp_files
  )
  
  # Print results
  print("\nProcessed files:")
  for data_type, path in results.items():
    print(f"  {data_type}: {path}")
  
  return 0


if __name__ == "__main__":
  sys.exit(main())
