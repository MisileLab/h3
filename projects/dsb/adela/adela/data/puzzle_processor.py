"""Module for processing puzzle data from Lichess database."""

import csv
import io
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Union, Tuple

from adela.core.chess_shim import chess
import polars as pl
from tqdm import tqdm

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PuzzleProcessor:
  """Processor for puzzle data from Lichess database."""

  def __init__(self, batch_size: int = 50000) -> None:
    """Initialize the processor.

    Args:
      batch_size: Number of puzzles to process in a batch.
    """
    self.batch_size = batch_size
  
  def _decompress_zstd(self, input_path: Path) -> Iterator[str]:
    """Decompress a zstd file and yield lines.

    Args:
      input_path: Path to the zstd file.

    Yields:
      Lines from the decompressed file.
    """
    # Use subprocess to decompress the file without creating a large temporary file
    process = subprocess.Popen(
      ["zstd", "-d", "-c", str(input_path)],
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
      text=True,
      bufsize=1
    )
    
    # Yield lines from the process stdout
    for line in process.stdout:
      yield line
    
    # Wait for the process to finish
    process.wait()
  
  def _parse_puzzle_csv_line(self, line: str) -> Dict[str, Any]:
    """Parse a line from the puzzle CSV file.

    Args:
      line: CSV line from the puzzle file.

    Returns:
      Dictionary with puzzle data.
    """
    # Parse CSV line
    reader = csv.reader([line])
    puzzle_id, fen, moves, rating, rating_deviation, popularity, nb_plays, themes, game_url, opening_tags = next(reader)
    
    # Convert moves to a list
    moves_list = moves.split()
    
    # Parse themes
    themes_list = themes.split()
    
    # Parse opening tags
    opening_tags_list = opening_tags.split() if opening_tags else []
    
    # Create puzzle record
    return {
      "puzzle_id": puzzle_id,
      "fen": fen,
      "moves": moves,
      "first_move": moves_list[0] if moves_list else "",
      "solution": " ".join(moves_list[1:]) if len(moves_list) > 1 else "",
      "rating": int(rating),
      "rating_deviation": int(rating_deviation),
      "popularity": int(popularity),
      "nb_plays": int(nb_plays),
      "themes": themes,
      "num_themes": len(themes_list),
      "has_mate": "mate" in themes,
      "is_short": "short" in themes,
      "game_url": game_url,
      "opening_tags": opening_tags,
      "num_opening_tags": len(opening_tags_list),
    }
  
  def process_puzzle_file(
    self, 
    puzzle_path: Path,
    output_dir: Optional[Path] = None,
    min_rating: int = 0,
    max_rating: Optional[int] = None,
    themes_filter: Optional[List[str]] = None,
    max_puzzles: Optional[int] = None
  ) -> List[Path]:
    """Process a puzzle CSV file and convert to Parquet.

    Args:
      puzzle_path: Path to the puzzle CSV file (can be zstd compressed).
      output_dir: Directory to save Parquet files. If None, uses the same directory as the CSV file.
      min_rating: Minimum puzzle rating to include.
      max_rating: Maximum puzzle rating to include. If None, no maximum.
      themes_filter: List of themes that puzzles must have at least one of. If None, no filter.
      max_puzzles: Maximum number of puzzles to process. If None, processes all puzzles.

    Returns:
      List of paths to the created Parquet files.
    """
    # Set output directory
    if output_dir is None:
      output_dir = puzzle_path.parent
    else:
      output_dir = Path(output_dir)
      output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine base output filename
    base_filename = puzzle_path.stem
    if base_filename.endswith('.csv'):
      base_filename = base_filename[:-4]
    
    logger.info(f"Processing puzzle file: {puzzle_path}")
    
    # Check if the file is zstd compressed
    is_zstd = puzzle_path.suffix.lower() == '.zst'
    
    # Initialize variables
    puzzles_processed = 0
    puzzles_included = 0
    batch_num = 0
    current_batch = []
    output_files = []
    
    # Open the file
    if is_zstd:
      # Use zstd decompression
      puzzle_iter = self._decompress_zstd(puzzle_path)
    else:
      # Regular file
      with open(puzzle_path, 'r', encoding='utf-8') as f:
        puzzle_iter = f
    
    # Skip header if present (first line starts with "PuzzleId")
    first_line = next(puzzle_iter, "")
    if not first_line.startswith("PuzzleId"):
      # Process the first line if it's not a header
      try:
        puzzle_record = self._parse_puzzle_csv_line(first_line)
        puzzles_processed += 1
        
        # Apply filters
        rating = puzzle_record["rating"]
        themes = puzzle_record["themes"].split()
        
        if (rating >= min_rating and 
            (max_rating is None or rating <= max_rating) and
            (themes_filter is None or any(theme in themes for theme in themes_filter))):
          current_batch.append(puzzle_record)
          puzzles_included += 1
      except Exception as e:
        logger.warning(f"Error processing puzzle: {e}")
    
    # Process the rest of the file
    for line in puzzle_iter:
      line = line.strip()
      if not line:
        continue
      
      try:
        puzzle_record = self._parse_puzzle_csv_line(line)
        puzzles_processed += 1
        
        # Apply filters
        rating = puzzle_record["rating"]
        themes = puzzle_record["themes"].split()
        
        if (rating >= min_rating and 
            (max_rating is None or rating <= max_rating) and
            (themes_filter is None or any(theme in themes for theme in themes_filter))):
          current_batch.append(puzzle_record)
          puzzles_included += 1
        
        # Process batch if it's full
        if len(current_batch) >= self.batch_size:
          # Convert to Polars DataFrame
          df = pl.DataFrame(current_batch)
          
          # Save to Parquet
          output_path = output_dir / f"{base_filename}_batch_{batch_num}.parquet"
          df.write_parquet(output_path)
          output_files.append(output_path)
          
          logger.info(f"Saved batch {batch_num} with {len(current_batch)} puzzles to {output_path}")
          
          # Reset batch
          current_batch = []
          batch_num += 1
        
        # Check if we've reached the maximum number of puzzles
        if max_puzzles is not None and puzzles_included >= max_puzzles:
          break
      
      except Exception as e:
        logger.warning(f"Error processing puzzle: {e}")
    
    # Process the final batch
    if current_batch:
      # Convert to Polars DataFrame
      df = pl.DataFrame(current_batch)
      
      # Save to Parquet
      output_path = output_dir / f"{base_filename}_batch_{batch_num}.parquet"
      df.write_parquet(output_path)
      output_files.append(output_path)
      
      logger.info(f"Saved final batch {batch_num} with {len(current_batch)} puzzles to {output_path}")
    
    logger.info(f"Processed {puzzles_processed} puzzles, included {puzzles_included}")
    
    return output_files
  
  def merge_parquet_files(
    self, 
    parquet_files: List[Path],
    output_path: Path
  ) -> Path:
    """Merge multiple Parquet files into one.

    Args:
      parquet_files: List of Parquet files to merge.
      output_path: Path to the output Parquet file.

    Returns:
      Path to the merged Parquet file.
    """
    logger.info(f"Merging {len(parquet_files)} Parquet files into {output_path}")
    
    # Read and concatenate all DataFrames
    dfs = [pl.read_parquet(file) for file in parquet_files]
    merged_df = pl.concat(dfs)
    
    # Save the merged DataFrame
    merged_df.write_parquet(output_path)
    
    logger.info(f"Merged {len(parquet_files)} files with {len(merged_df)} total rows")
    
    return output_path
