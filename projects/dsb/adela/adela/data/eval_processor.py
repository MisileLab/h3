"""Module for processing evaluation data from Lichess database."""

import json
import io
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Union, Tuple

import chess
import polars as pl
from tqdm import tqdm

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationProcessor:
  """Processor for evaluation data from Lichess database."""

  def __init__(self, batch_size: int = 50000) -> None:
    """Initialize the processor.

    Args:
      batch_size: Number of evaluations to process in a batch.
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
  
  def _parse_evaluation_line(self, line: str) -> Dict[str, Any]:
    """Parse a line from the evaluation JSONL file.

    Args:
      line: JSON line from the evaluation file.

    Returns:
      Dictionary with evaluation data.
    """
    # Parse JSON line
    eval_data = json.loads(line)
    
    # Get the FEN
    fen = eval_data.get("fen", "")
    
    # Get the best evaluation (highest depth)
    best_eval = max(eval_data.get("evals", []), key=lambda x: x.get("depth", 0), default={})
    
    # Get the first PV (principal variation)
    pvs = best_eval.get("pvs", [])
    first_pv = pvs[0] if pvs else {}
    
    # Get evaluation value
    cp = first_pv.get("cp")
    mate = first_pv.get("mate")
    
    # Convert mate score to centipawns
    if mate is not None:
      if mate > 0:
        eval_value = 10000 - mate  # Positive mate
      else:
        eval_value = -10000 - mate  # Negative mate
    else:
      eval_value = cp
    
    # Get the line (moves)
    line = first_pv.get("line", "")
    
    # Create evaluation record
    return {
      "fen": fen,
      "depth": best_eval.get("depth"),
      "knodes": best_eval.get("knodes"),
      "eval_cp": cp,
      "eval_mate": mate,
      "eval_value": eval_value,
      "line": line,
      "num_pvs": len(pvs),
      "has_mate": mate is not None,
    }
  
  def process_evaluation_file(
    self, 
    eval_path: Path,
    output_dir: Optional[Path] = None,
    min_depth: int = 0,
    max_evaluations: Optional[int] = None
  ) -> List[Path]:
    """Process an evaluation JSONL file and convert to Parquet.

    Args:
      eval_path: Path to the evaluation JSONL file (can be zstd compressed).
      output_dir: Directory to save Parquet files. If None, uses the same directory as the JSONL file.
      min_depth: Minimum evaluation depth to include.
      max_evaluations: Maximum number of evaluations to process. If None, processes all evaluations.

    Returns:
      List of paths to the created Parquet files.
    """
    # Set output directory
    if output_dir is None:
      output_dir = eval_path.parent
    else:
      output_dir = Path(output_dir)
      output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine base output filename
    base_filename = eval_path.stem
    if base_filename.endswith('.jsonl'):
      base_filename = base_filename[:-6]
    
    logger.info(f"Processing evaluation file: {eval_path}")
    
    # Check if the file is zstd compressed
    is_zstd = eval_path.suffix.lower() == '.zst'
    
    # Initialize variables
    evals_processed = 0
    evals_included = 0
    batch_num = 0
    current_batch = []
    output_files = []
    
    # Open the file
    if is_zstd:
      # Use zstd decompression
      eval_iter = self._decompress_zstd(eval_path)
    else:
      # Regular file
      with open(eval_path, 'r', encoding='utf-8') as f:
        eval_iter = f
    
    # Process the file
    for line in eval_iter:
      line = line.strip()
      if not line:
        continue
      
      try:
        eval_record = self._parse_evaluation_line(line)
        evals_processed += 1
        
        # Apply filters
        depth = eval_record.get("depth", 0)
        
        if depth >= min_depth:
          current_batch.append(eval_record)
          evals_included += 1
        
        # Process batch if it's full
        if len(current_batch) >= self.batch_size:
          # Convert to Polars DataFrame
          df = pl.DataFrame(current_batch)
          
          # Save to Parquet
          output_path = output_dir / f"{base_filename}_batch_{batch_num}.parquet"
          df.write_parquet(output_path)
          output_files.append(output_path)
          
          logger.info(f"Saved batch {batch_num} with {len(current_batch)} evaluations to {output_path}")
          
          # Reset batch
          current_batch = []
          batch_num += 1
        
        # Check if we've reached the maximum number of evaluations
        if max_evaluations is not None and evals_included >= max_evaluations:
          break
      
      except Exception as e:
        logger.warning(f"Error processing evaluation: {e}")
    
    # Process the final batch
    if current_batch:
      # Convert to Polars DataFrame
      df = pl.DataFrame(current_batch)
      
      # Save to Parquet
      output_path = output_dir / f"{base_filename}_batch_{batch_num}.parquet"
      df.write_parquet(output_path)
      output_files.append(output_path)
      
      logger.info(f"Saved final batch {batch_num} with {len(current_batch)} evaluations to {output_path}")
    
    logger.info(f"Processed {evals_processed} evaluations, included {evals_included}")
    
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
