"""Pipeline for downloading and processing Lichess data."""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

import polars as pl

from adela.data.downloader import LichessDownloader
from adela.data.pgn_parser import PGNParser
from adela.data.puzzle_processor import PuzzleProcessor
from adela.data.eval_processor import EvaluationProcessor

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LichessDataPipeline:
  """Pipeline for downloading and processing Lichess data."""

  def __init__(
    self, 
    data_dir: Optional[Union[str, Path]] = None,
    temp_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 50000
  ) -> None:
    """Initialize the pipeline.

    Args:
      data_dir: Directory to store processed data. If None, uses './data'.
      temp_dir: Directory to store temporary files. If None, uses system temp directory.
      batch_size: Number of records to process in a batch.
    """
    # Set data directory
    if data_dir is None:
      self.data_dir = Path("./data")
    else:
      self.data_dir = Path(data_dir)
    
    # Create data directory if it doesn't exist
    self.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Set temp directory
    if temp_dir is None:
      self.temp_dir = Path(tempfile.gettempdir()) / "adela_lichess_temp"
    else:
      self.temp_dir = Path(temp_dir)
    
    # Create temp directory if it doesn't exist
    self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each data type
    for data_type in ["games", "puzzles", "evaluations"]:
      (self.data_dir / data_type).mkdir(exist_ok=True)
      (self.temp_dir / data_type).mkdir(exist_ok=True)
    
    # Initialize components
    self.downloader = LichessDownloader(self.temp_dir)
    self.pgn_parser = PGNParser(batch_size=batch_size)
    self.puzzle_processor = PuzzleProcessor(batch_size=batch_size)
    self.eval_processor = EvaluationProcessor(batch_size=batch_size)
    
    logger.info(f"Data pipeline initialized with data_dir={self.data_dir}, temp_dir={self.temp_dir}")
  
  def process_games(
    self, 
    num_months: int = 1,
    min_elo: int = 1800,
    max_games: Optional[int] = None,
    delete_temp_files: bool = True
  ) -> Path:
    """Download and process chess games.

    Args:
      num_months: Number of months of games to download.
      min_elo: Minimum Elo rating for games to include.
      max_games: Maximum number of games to process. If None, processes all games.
      delete_temp_files: Whether to delete temporary files after processing.

    Returns:
      Path to the processed Parquet file.
    """
    logger.info(f"Processing {num_months} months of games with min_elo={min_elo}")
    
    # Download the latest game files
    downloaded_files = self.downloader.download_latest_data("games", num_months=num_months)
    
    # Process each file
    all_parquet_files = []
    for pgn_file in downloaded_files:
      # Parse the PGN file
      parquet_files = self.pgn_parser.parse_pgn_file(
        pgn_file,
        output_dir=self.temp_dir / "games",
        min_elo=min_elo,
        max_games=max_games
      )
      all_parquet_files.extend(parquet_files)
    
    # Merge all Parquet files
    output_filename = f"lichess_games_{datetime.now().strftime('%Y%m%d')}.parquet"
    output_path = self.data_dir / "games" / output_filename
    
    merged_path = self.pgn_parser.merge_parquet_files(all_parquet_files, output_path)
    
    # Delete temporary files if requested
    if delete_temp_files:
      for file in all_parquet_files:
        file.unlink()
      logger.info(f"Deleted {len(all_parquet_files)} temporary files")
    
    logger.info(f"Processed games saved to {merged_path}")
    
    return merged_path
  
  def process_puzzles(
    self, 
    min_rating: int = 0,
    max_rating: Optional[int] = None,
    themes_filter: Optional[List[str]] = None,
    max_puzzles: Optional[int] = None,
    delete_temp_files: bool = True
  ) -> Path:
    """Download and process chess puzzles.

    Args:
      min_rating: Minimum puzzle rating to include.
      max_rating: Maximum puzzle rating to include. If None, no maximum.
      themes_filter: List of themes that puzzles must have at least one of. If None, no filter.
      max_puzzles: Maximum number of puzzles to process. If None, processes all puzzles.
      delete_temp_files: Whether to delete temporary files after processing.

    Returns:
      Path to the processed Parquet file.
    """
    logger.info(f"Processing puzzles with min_rating={min_rating}, max_rating={max_rating}")
    
    # Download the puzzle file
    downloaded_files = self.downloader.download_latest_data("puzzles")
    
    if not downloaded_files:
      logger.error("No puzzle files downloaded")
      return None
    
    # Process the puzzle file
    puzzle_file = downloaded_files[0]
    parquet_files = self.puzzle_processor.process_puzzle_file(
      puzzle_file,
      output_dir=self.temp_dir / "puzzles",
      min_rating=min_rating,
      max_rating=max_rating,
      themes_filter=themes_filter,
      max_puzzles=max_puzzles
    )
    
    # Merge all Parquet files
    output_filename = f"lichess_puzzles_{datetime.now().strftime('%Y%m%d')}.parquet"
    output_path = self.data_dir / "puzzles" / output_filename
    
    merged_path = self.puzzle_processor.merge_parquet_files(parquet_files, output_path)
    
    # Delete temporary files if requested
    if delete_temp_files:
      for file in parquet_files:
        file.unlink()
      logger.info(f"Deleted {len(parquet_files)} temporary files")
    
    logger.info(f"Processed puzzles saved to {merged_path}")
    
    return merged_path
  
  def process_evaluations(
    self, 
    min_depth: int = 20,
    max_evaluations: Optional[int] = None,
    delete_temp_files: bool = True
  ) -> Path:
    """Download and process chess position evaluations.

    Args:
      min_depth: Minimum evaluation depth to include.
      max_evaluations: Maximum number of evaluations to process. If None, processes all evaluations.
      delete_temp_files: Whether to delete temporary files after processing.

    Returns:
      Path to the processed Parquet file.
    """
    logger.info(f"Processing evaluations with min_depth={min_depth}")
    
    # Download the evaluation file
    downloaded_files = self.downloader.download_latest_data("evaluations")
    
    if not downloaded_files:
      logger.error("No evaluation files downloaded")
      return None
    
    # Process the evaluation file
    eval_file = downloaded_files[0]
    parquet_files = self.eval_processor.process_evaluation_file(
      eval_file,
      output_dir=self.temp_dir / "evaluations",
      min_depth=min_depth,
      max_evaluations=max_evaluations
    )
    
    # Merge all Parquet files
    output_filename = f"lichess_evaluations_{datetime.now().strftime('%Y%m%d')}.parquet"
    output_path = self.data_dir / "evaluations" / output_filename
    
    merged_path = self.eval_processor.merge_parquet_files(parquet_files, output_path)
    
    # Delete temporary files if requested
    if delete_temp_files:
      for file in parquet_files:
        file.unlink()
      logger.info(f"Deleted {len(parquet_files)} temporary files")
    
    logger.info(f"Processed evaluations saved to {merged_path}")
    
    return merged_path
  
  def run_full_pipeline(
    self,
    process_games: bool = True,
    process_puzzles: bool = True,
    process_evaluations: bool = True,
    num_months: int = 1,
    min_elo: int = 1800,
    max_games: Optional[int] = None,
    min_puzzle_rating: int = 0,
    max_puzzle_rating: Optional[int] = None,
    puzzle_themes: Optional[List[str]] = None,
    max_puzzles: Optional[int] = None,
    min_eval_depth: int = 20,
    max_evaluations: Optional[int] = None,
    delete_temp_files: bool = True
  ) -> Dict[str, Path]:
    """Run the full data pipeline.

    Args:
      process_games: Whether to process chess games.
      process_puzzles: Whether to process chess puzzles.
      process_evaluations: Whether to process chess position evaluations.
      num_months: Number of months of games to download.
      min_elo: Minimum Elo rating for games to include.
      max_games: Maximum number of games to process. If None, processes all games.
      min_puzzle_rating: Minimum puzzle rating to include.
      max_puzzle_rating: Maximum puzzle rating to include. If None, no maximum.
      puzzle_themes: List of themes that puzzles must have at least one of. If None, no filter.
      max_puzzles: Maximum number of puzzles to process. If None, processes all puzzles.
      min_eval_depth: Minimum evaluation depth to include.
      max_evaluations: Maximum number of evaluations to process. If None, processes all evaluations.
      delete_temp_files: Whether to delete temporary files after processing.

    Returns:
      Dictionary mapping data types to paths of processed Parquet files.
    """
    results = {}
    
    # Process games
    if process_games:
      games_path = self.process_games(
        num_months=num_months,
        min_elo=min_elo,
        max_games=max_games,
        delete_temp_files=delete_temp_files
      )
      results["games"] = games_path
    
    # Process puzzles
    if process_puzzles:
      puzzles_path = self.process_puzzles(
        min_rating=min_puzzle_rating,
        max_rating=max_puzzle_rating,
        themes_filter=puzzle_themes,
        max_puzzles=max_puzzles,
        delete_temp_files=delete_temp_files
      )
      results["puzzles"] = puzzles_path
    
    # Process evaluations
    if process_evaluations:
      evals_path = self.process_evaluations(
        min_depth=min_eval_depth,
        max_evaluations=max_evaluations,
        delete_temp_files=delete_temp_files
      )
      results["evaluations"] = evals_path
    
    logger.info("Full pipeline completed")
    
    return results
