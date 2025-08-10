"""Module for parsing PGN files from Lichess database."""

import re
import io
import os
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Union, Tuple

from adela.core.chess_shim import chess, require_pgn
import polars as pl
from tqdm import tqdm

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PGNParser:
  """Parser for PGN files from Lichess database."""

  def __init__(self, batch_size: int = 10000) -> None:
    """Initialize the parser.

    Args:
      batch_size: Number of games to process in a batch.
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
  
  def _extract_game_headers(self, pgn_text: str) -> Dict[str, Any]:
    """Extract headers from a PGN game.

    Args:
      pgn_text: PGN text of a game.

    Returns:
      Dictionary with game headers.
    """
    headers = {}
    
    # Extract headers using regex
    header_pattern = r'\[(\w+)\s+"([^"]*)"\]'
    for match in re.finditer(header_pattern, pgn_text):
      key, value = match.groups()
      headers[key] = value
    
    return headers
  
  def _extract_moves_and_evals(self, pgn_text: str) -> Tuple[List[str], List[float]]:
    """Extract moves and evaluations from a PGN game.

    Args:
      pgn_text: PGN text of a game.

    Returns:
      Tuple of (moves, evaluations).
    """
    # Extract the moves section (after the headers)
    moves_section = re.sub(r'\[.*?\]\s*', '', pgn_text).strip()
    
    # Extract moves and evaluations
    moves = []
    evals = []
    
    # Extract moves and evaluations using regex
    move_eval_pattern = r'(\d+\.\s+)?([\w\-=+#]+)(\s+\{[^\}]*\})?'
    for match in re.finditer(move_eval_pattern, moves_section):
      move_num, move, eval_comment = match.groups()
      
      if move and move not in ["1-0", "0-1", "1/2-1/2", "*"]:
        moves.append(move)
        
        # Extract evaluation if available
        if eval_comment:
          eval_match = re.search(r'\[%eval\s+([-+]?\d+\.\d+|#[-+]?\d+)\]', eval_comment)
          if eval_match:
            eval_str = eval_match.group(1)
            if eval_str.startswith('#'):
              # Convert mate score to a large value
              mate_in = int(eval_str[1:])
              eval_value = 10000 if mate_in > 0 else -10000
            else:
              eval_value = float(eval_str)
            evals.append(eval_value)
          else:
            evals.append(None)
        else:
          evals.append(None)
    
    return moves, evals
  
  def parse_pgn_file(
    self, 
    pgn_path: Path,
    output_dir: Optional[Path] = None,
    min_elo: int = 0,
    max_games: Optional[int] = None
  ) -> List[Path]:
    """Parse a PGN file and convert to Parquet.

    Args:
      pgn_path: Path to the PGN file (can be zstd compressed).
      output_dir: Directory to save Parquet files. If None, uses the same directory as the PGN file.
      min_elo: Minimum Elo rating for games to include.
      max_games: Maximum number of games to process. If None, processes all games.

    Returns:
      List of paths to the created Parquet files.
    """
    # Ensure PGN parser is available if needed in future extensions
    _ = require_pgn()

    # Set output directory
    if output_dir is None:
      output_dir = pgn_path.parent
    else:
      output_dir = Path(output_dir)
      output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine base output filename
    base_filename = pgn_path.stem
    if base_filename.endswith('.pgn'):
      base_filename = base_filename[:-4]
    
    logger.info(f"Parsing PGN file: {pgn_path}")
    
    # Check if the file is zstd compressed
    is_zstd = pgn_path.suffix.lower() == '.zst'
    
    # Initialize variables
    games_processed = 0
    batch_num = 0
    current_batch = []
    output_files = []
    
    # Open the file
    if is_zstd:
      # Use zstd decompression
      pgn_iter = self._decompress_zstd(pgn_path)
    else:
      # Regular file
      with open(pgn_path, 'r', encoding='utf-8') as f:
        pgn_iter = f
    
    # Process the file
    current_game = []
    for line in pgn_iter:
      line = line.strip()
      
      # Collect lines for the current game
      if line or current_game:
        current_game.append(line)
      
      # Empty line after a game indicates the end of a game
      if not line and current_game:
        pgn_text = '\n'.join(current_game)
        current_game = []
        
        # Extract game headers
        headers = self._extract_game_headers(pgn_text)
        
        # Check Elo ratings
        white_elo = int(headers.get('WhiteElo', '0'))
        black_elo = int(headers.get('BlackElo', '0'))
        
        if white_elo >= min_elo and black_elo >= min_elo:
          # Extract moves and evaluations
          moves, evals = self._extract_moves_and_evals(pgn_text)
          
          # Create game record
          game_record = {
            'id': headers.get('Site', '').split('/')[-1],
            'white': headers.get('White', ''),
            'black': headers.get('Black', ''),
            'white_elo': white_elo,
            'black_elo': black_elo,
            'result': headers.get('Result', '*'),
            'date': headers.get('UTCDate', headers.get('Date', '')),
            'time': headers.get('UTCTime', ''),
            'time_control': headers.get('TimeControl', ''),
            'eco': headers.get('ECO', ''),
            'opening': headers.get('Opening', ''),
            'termination': headers.get('Termination', ''),
            'moves': ' '.join(moves),
            'num_moves': len(moves),
            'has_evals': any(e is not None for e in evals),
            'avg_eval': sum(e for e in evals if e is not None) / len([e for e in evals if e is not None]) if any(e is not None for e in evals) else None,
            'min_eval': min((e for e in evals if e is not None), default=None),
            'max_eval': max((e for e in evals if e is not None), default=None),
          }
          
          current_batch.append(game_record)
          games_processed += 1
        
        # Process batch if it's full
        if len(current_batch) >= self.batch_size:
          # Convert to Polars DataFrame
          df = pl.DataFrame(current_batch)
          
          # Save to Parquet
          output_path = output_dir / f"{base_filename}_batch_{batch_num}.parquet"
          df.write_parquet(output_path)
          output_files.append(output_path)
          
          logger.info(f"Saved batch {batch_num} with {len(current_batch)} games to {output_path}")
          
          # Reset batch
          current_batch = []
          batch_num += 1
        
        # Check if we've reached the maximum number of games
        if max_games is not None and games_processed >= max_games:
          break
    
    # Process the final batch
    if current_batch:
      # Convert to Polars DataFrame
      df = pl.DataFrame(current_batch)
      
      # Save to Parquet
      output_path = output_dir / f"{base_filename}_batch_{batch_num}.parquet"
      df.write_parquet(output_path)
      output_files.append(output_path)
      
      logger.info(f"Saved final batch {batch_num} with {len(current_batch)} games to {output_path}")
    
    logger.info(f"Processed {games_processed} games in total")
    
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
