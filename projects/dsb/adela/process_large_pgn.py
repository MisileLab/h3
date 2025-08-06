#!/usr/bin/env python
"""Script to process very large PGN files (200GB+) line by line and save to Parquet."""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

from adela.data.large_pgn_parser import LargePGNParser

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  handlers=[
    logging.StreamHandler(),
    logging.FileHandler(f"pgn_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
  ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments.

  Returns:
    Parsed arguments.
  """
  parser = argparse.ArgumentParser(description="Process very large PGN files line by line and save to Parquet")
  
  parser.add_argument(
    "pgn_file",
    type=str,
    help="Path to the PGN file to process"
  )
  
  parser.add_argument(
    "--output-dir",
    type=str,
    default="./data/games",
    help="Directory to save Parquet files"
  )
  
  parser.add_argument(
    "--batch-size",
    type=int,
    default=50000,
    help="Number of games to process in a batch"
  )
  
  parser.add_argument(
    "--min-elo",
    type=int,
    default=1000,
    help="Minimum Elo rating for games to include"
  )
  
  parser.add_argument(
    "--output-prefix",
    type=str,
    default=f"chess_games_{datetime.now().strftime('%Y%m%d')}",
    help="Prefix for output Parquet filenames"
  )
  
  parser.add_argument(
    "--no-merge",
    action="store_true",
    help="Don't merge batch files into a single Parquet file"
  )
  
  parser.add_argument(
    "--no-progress",
    action="store_true",
    help="Don't show progress bar"
  )
  
  parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="Number of worker threads to use for parallel processing (default: 0 = use all CPU cores)"
  )
  
  return parser.parse_args()


def main() -> int:
  """Main entry point.

  Returns:
    Exit code.
  """
  args = parse_args()
  
  # Check if the PGN file exists
  pgn_path = Path(args.pgn_file)
  if not pgn_path.exists():
    logger.error(f"PGN file not found: {pgn_path}")
    return 1
  
  # Create output directory
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  
  logger.info(f"Processing PGN file: {pgn_path}")
  logger.info(f"Output directory: {output_dir}")
  logger.info(f"Batch size: {args.batch_size}")
  logger.info(f"Minimum Elo: {args.min_elo}")
  logger.info(f"Number of workers: {args.num_workers}")
  
  try:
    # Create parser
    parser = LargePGNParser(
      batch_size=args.batch_size,
      output_dir=output_dir,
      min_elo=args.min_elo,
      output_filename_prefix=args.output_prefix,
      num_workers=args.num_workers
    )
    
    # Start timing
    start_time = datetime.now()
    logger.info(f"Started processing at {start_time}")
    
    # Parse the PGN file
    games_processed = parser.parse_pgn_file(
      pgn_path=pgn_path,
      show_progress=not args.no_progress
    )
    
    # Merge batch files if requested
    if not args.no_merge:
      logger.info("Merging batch files...")
      merged_path = parser.merge_parquet_files()
      logger.info(f"Merged file saved to {merged_path}")
    
    # End timing
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Finished processing at {end_time}")
    logger.info(f"Total duration: {duration}")
    logger.info(f"Processed {games_processed} games")
    
    return 0
  
  except Exception as e:
    logger.exception(f"Error processing PGN file: {e}")
    return 1


if __name__ == "__main__":
  sys.exit(main())
