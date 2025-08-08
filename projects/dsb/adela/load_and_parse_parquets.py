#!/usr/bin/env python3
"""Script to load Parquet files, filter by Elo, and parse movetext to chess moves."""

import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import logging
import gc
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import polars as pl
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_movetext_to_moves(movetext: str) -> list[str]:
    """Parse a movetext string to extract individual chess moves in SAN notation.
    
    Example movetext:
    '1. e4 e6 2. d4 b6 3. a3 Bb7 4. Nc3 Nh6 5. Bxh6 gxh6 6. Be2 Qg5 7. Bg4 h5 8. Nf3 Qg6 9. Nh4 Qg5 10. Bxh5 Qxh4 11. Qf3 Kd8 12. Qxf7 Nc6 13. Qe8# 1-0'
    
    Args:
        movetext: PGN movetext string containing the game moves
        
    Returns:
        List of chess moves in SAN notation
    """
    if not movetext:
        return []
    
    # Remove result markers and normalize whitespace
    cleaned = re.sub(r'(1-0|0-1|1/2-1/2|\*)\s*$', '', movetext)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    moves = []
    
    # Simple but effective approach: split on move numbers and extract moves
    # This handles most PGN formats reliably
    tokens = re.split(r'\d+\.', cleaned)
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        
        # Extract moves from each token using a comprehensive pattern
        # This pattern matches most chess moves in SAN notation
        token_moves = re.findall(
            r'\b(?:[KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBNP])?[+#]?|O-O(?:-O)?[+#]?)\b',
            token
        )
        
        for move in token_moves:
            # Filter out result markers and invalid moves
            if (move and 
                move not in {'1-0', '0-1', '1/2-1/2', '*'} and
                len(move) >= 2 and
                not move.isdigit()):
                moves.append(move)
    
    return moves





def _process_single_file(
    file_info: tuple[int, Path, int],
    min_elo: int,
    file_chunk_size: int,
    parse_chunk_size: int,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[pl.DataFrame]:
    """Process a single Parquet file and return chunks of processed data.
    
    Args:
        file_info: Tuple of (file_index, parquet_file_path, total_files)
        min_elo: Minimum Elo rating threshold
        file_chunk_size: Number of games to read at once
        parse_chunk_size: Number of games to parse movetext for at once
        
    Returns:
        List of processed DataFrame chunks
    """
    file_idx, parquet_file, total_files = file_info
    
    logger.info(f"Processing file {file_idx + 1}/{total_files}: {parquet_file.name}")
    
    try:
        processed_chunks = []
        
        # Use scan_parquet to get total row count first
        total_rows = pl.scan_parquet(str(parquet_file)).select(pl.count()).collect().item()
        
        for chunk_start in range(0, total_rows, file_chunk_size):
            # Read chunk with Elo filtering
            chunk_df = pl.scan_parquet(str(parquet_file)).slice(
                chunk_start, file_chunk_size
            ).filter(
                (pl.col("WhiteElo") >= min_elo) | (pl.col("BlackElo") >= min_elo)
            ).collect()
            
            if len(chunk_df) == 0:
                # Update processed count even if nothing included
                if progress_callback is not None:
                    rows_in_slice = min(file_chunk_size, max(0, total_rows - chunk_start))
                    progress_callback(rows_in_slice, 0)
                continue
            
            # Parse movetext efficiently in smaller sub-chunks
            parsed_chunk = add_parsed_moves(chunk_df, chunk_size=parse_chunk_size)
            processed_chunks.append(parsed_chunk)

            # Update progress with processed and included counts
            if progress_callback is not None:
                rows_in_slice = min(file_chunk_size, max(0, total_rows - chunk_start))
                progress_callback(rows_in_slice, len(parsed_chunk))
            
            # Clean up memory
            del chunk_df
            gc.collect()
        
        logger.info(f"Completed file {file_idx + 1}/{total_files}: {parquet_file.name} - {len(processed_chunks)} chunks")
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error processing file {parquet_file}: {e}")
        return []


def process_parquet_files_in_batches(
    folder_path: Path, 
    min_elo: int = 1500, 
    batch_size: int = 50000,
    output_dir: Optional[Path] = None,
    output_prefix: str = "processed_games",
    file_chunk_size: int = 5000,
    parse_chunk_size: int = 500,
    num_threads: int = 0
) -> List[Path]:
    """Process Parquet files in batches to handle large datasets efficiently.
    
    Args:
        folder_path: Path to folder containing Parquet files
        min_elo: Minimum Elo rating threshold
        batch_size: Number of games per output batch (default: 50000)
        output_dir: Directory to save processed batches (default: current directory)
        output_prefix: Prefix for output filenames
        file_chunk_size: Number of games to read from each file at once (default: 5000)
        parse_chunk_size: Number of games to parse movetext for at once (default: 500)
        num_threads: Number of worker threads (0 = auto-detect, default: 0)
        
    Returns:
        List of paths to created batch files
        
    Raises:
        ValueError: If no Parquet files found or folder doesn't exist
    """
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Find all Parquet files in the folder
    parquet_files = sorted(folder_path.glob("*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No Parquet files found in folder: {folder_path}")
    
    if output_dir is None:
        output_dir = Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Found {len(parquet_files)} Parquet files in {folder_path}")
    logger.info(f"Processing in batches of {batch_size} games")
    
    # Determine optimal number of threads
    if num_threads <= 0:
        num_threads = min(os.cpu_count() or 1, len(parquet_files), 4)
    
    logger.info(f"Using {num_threads} threads for parallel processing")
    
    # Precompute total number of rows for progress bar
    file_to_total_rows: Dict[Path, int] = {}
    for f in parquet_files:
        try:
            file_to_total_rows[f] = pl.scan_parquet(str(f)).select(pl.count()).collect().item()
        except Exception:
            file_to_total_rows[f] = 0
    grand_total_rows = sum(file_to_total_rows.values())

    # Initialize batch tracking with thread safety
    current_batch_data = []
    batch_num = 0
    total_processed = 0
    total_included = 0
    output_files = []
    batch_lock = threading.Lock()
    progress_lock = threading.Lock()

    # Global games progress bar
    games_pbar = tqdm(total=grand_total_rows, desc="Processing games")

    def progress_callback(processed_rows: int, included_rows: int) -> None:
        nonlocal total_processed, total_included
        with progress_lock:
            total_processed += processed_rows
            total_included += included_rows
            # Ensure bar reflects processed rows
            games_pbar.update(processed_rows)
            # Show included as postfix
            games_pbar.set_postfix(included=total_included)
    
    def save_batch_if_needed():
        """Thread-safe batch saving function."""
        nonlocal current_batch_data, batch_num, output_files
        
        with batch_lock:
            current_total = sum(len(chunk) for chunk in current_batch_data)
            if current_total >= batch_size:
                # Combine chunks and save batch
                combined_df = pl.concat(current_batch_data)
                
                # Save exactly batch_size games
                batch_df = combined_df.head(batch_size)
                output_file = _save_batch_df(batch_df, batch_num, output_dir, output_prefix)
                if output_file:
                    output_files.append(output_file)
                
                # Keep remaining games for next batch
                remaining_games = combined_df.tail(current_total - batch_size) if current_total > batch_size else None
                current_batch_data = [remaining_games] if remaining_games is not None and len(remaining_games) > 0 else []
                
                batch_num += 1
                logger.info(f"Saved batch {batch_num} with {batch_size} games")
                
                # Force garbage collection
                del combined_df, batch_df
                gc.collect()
    
    # Prepare file info for threading
    file_infos = [(i, file, len(parquet_files)) for i, file in enumerate(parquet_files)]
    
    # Process files using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all file processing tasks
        future_to_file = {
            executor.submit(_process_single_file, file_info, min_elo, file_chunk_size, parse_chunk_size, progress_callback): file_info
            for file_info in file_infos
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(parquet_files), desc="Processing files") as pbar:
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                file_idx, parquet_file, _ = file_info
                
                try:
                    processed_chunks = future.result()
                    
                    # Add processed chunks to batch (thread-safe)
                    with batch_lock:
                        current_batch_data.extend(processed_chunks)
                    
                    # Check if we need to save a batch
                    save_batch_if_needed()
                    
                except Exception as e:
                    logger.error(f"Error processing file {parquet_file}: {e}")
                
                pbar.update(1)
                gc.collect()
    
    # Save final batch if it has any games
    if current_batch_data:
        combined_df = pl.concat(current_batch_data)
        if len(combined_df) > 0:
            output_file = _save_batch_df(combined_df, batch_num, output_dir, output_prefix)
            if output_file:
                output_files.append(output_file)
            logger.info(f"Saved final batch {batch_num + 1} with {len(combined_df)} games")
    
    # Close progress bar
    games_pbar.close()

    logger.info(f"Processing complete: {total_processed} games processed, {total_included} games included")
    logger.info(f"Created {len(output_files)} batch files in {output_dir}")
    
    return output_files


def _save_batch_df(batch_df: pl.DataFrame, batch_num: int, output_dir: Path, prefix: str) -> Optional[Path]:
    """Save a batch DataFrame to a Parquet file.
    
    Args:
        batch_df: DataFrame containing processed games
        batch_num: Batch number for filename
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Path to saved file
    """
    if batch_df is None or len(batch_df) == 0:
        return None
    
    # Generate output filename
    output_file = output_dir / f"{prefix}_batch_{batch_num:06d}.parquet"
    
    # Save with compression
    batch_df.write_parquet(
        output_file,
        compression="zstd",
        use_pyarrow=True
    )
    
    return output_file


def _save_batch(batch_data: List[Dict[str, Any]], batch_num: int, output_dir: Path, prefix: str) -> Optional[Path]:
    """Save a batch of processed games to a Parquet file.
    
    Args:
        batch_data: List of game dictionaries
        batch_num: Batch number for filename
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Path to saved file
    """
    if not batch_data:
        return None
    
    # Create DataFrame from batch
    batch_df = pl.DataFrame(batch_data)
    
    return _save_batch_df(batch_df, batch_num, output_dir, prefix)


def load_parquet_files(folder_path: Path, max_memory_mb: int = 2048) -> pl.DataFrame:
    """Load all Parquet files from a folder and combine them with memory management.
    
    NOTE: For datasets >5GB, consider using process_parquet_files_in_batches() instead.
    
    Args:
        folder_path: Path to folder containing Parquet files
        max_memory_mb: Maximum memory usage in MB before warning (default: 2048MB)
        
    Returns:
        Combined Polars DataFrame
        
    Raises:
        ValueError: If no Parquet files found or folder doesn't exist
    """
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Find all Parquet files in the folder
    parquet_files = list(folder_path.glob("*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No Parquet files found in folder: {folder_path}")
    
    logger.info(f"Found {len(parquet_files)} Parquet files in {folder_path}")
    
    # Estimate memory usage
    total_size_mb = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)
    if total_size_mb > max_memory_mb:
        logger.warning(f"Dataset size ({total_size_mb:.1f}MB) exceeds recommended limit ({max_memory_mb}MB)")
        logger.warning("Consider using --batch-mode for better memory efficiency")
    
    # Load files efficiently
    if len(parquet_files) == 1:
        df = pl.read_parquet(parquet_files[0])
    else:
        # Load files in groups to manage memory
        avg_file_size_mb = total_size_mb / len(parquet_files) if len(parquet_files) > 0 else 1
        max_files_per_group = max(1, min(10, max_memory_mb // max(1, int(avg_file_size_mb))))
        
        all_dfs = []
        for i in range(0, len(parquet_files), max_files_per_group):
            group_files = parquet_files[i:i + max_files_per_group]
            logger.debug(f"Loading file group {i//max_files_per_group + 1}: {len(group_files)} files")
            
            # Use scan_parquet for efficient lazy loading
            lazy_dfs = [pl.scan_parquet(str(file)) for file in group_files]
            group_df = pl.concat(lazy_dfs).collect()
            all_dfs.append(group_df)
            
            # Force garbage collection between groups
            gc.collect()
        
        # Combine all groups
        logger.info("Combining all file groups...")
        df = pl.concat(all_dfs)
        
        # Clean up intermediate dataframes
        del all_dfs
        gc.collect()
    
    logger.info(f"Loaded {len(df)} total games from Parquet files")
    return df


def filter_by_elo(df: pl.DataFrame, min_elo: int = 1500) -> pl.DataFrame:
    """Filter games where either WhiteElo >= min_elo or BlackElo >= min_elo.
    
    Args:
        df: DataFrame containing chess games
        min_elo: Minimum Elo rating threshold
        
    Returns:
        Filtered DataFrame
    """
    original_count = len(df)
    
    # Filter where either white or black player has Elo >= min_elo
    filtered_df = df.filter(
        (pl.col("WhiteElo") >= min_elo) | (pl.col("BlackElo") >= min_elo)
    )
    
    filtered_count = len(filtered_df)
    logger.info(f"Filtered from {original_count} to {filtered_count} games with Elo >= {min_elo}")
    
    return filtered_df


def add_parsed_moves(df: pl.DataFrame, chunk_size: int = 1000) -> pl.DataFrame:
    """Add a column with parsed chess moves from movetext using memory-efficient chunked processing.
    
    Args:
        df: DataFrame containing chess games with movetext column
        chunk_size: Number of rows to process at once (default: 1000)
        
    Returns:
        DataFrame with additional 'parsed_moves' column
    """
    logger.info(f"Parsing movetext to extract chess moves (chunk size: {chunk_size})...")
    
    # Check if movetext column exists
    if "movetext" not in df.columns:
        # Check for 'moves' column as alternative
        if "moves" in df.columns:
            movetext_col = "moves"
        else:
            available_cols = [col for col in df.columns if 'move' in col.lower()]
            if available_cols:
                movetext_col = available_cols[0]
                logger.warning(f"Using column '{movetext_col}' as movetext source")
            else:
                raise ValueError("No movetext column found. Available columns: " + ", ".join(df.columns))
    else:
        movetext_col = "movetext"
    
    # Process in chunks to avoid memory issues
    total_rows = len(df)
    processed_chunks = []
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk_df = df.slice(start_idx, end_idx - start_idx)
        
        # Process this chunk
        chunk_with_moves = chunk_df.with_columns([
            pl.col(movetext_col).map_elements(
                parse_movetext_to_moves, 
                return_dtype=pl.List(pl.Utf8)
            ).alias("parsed_moves")
        ])
        
        # Add move count
        chunk_with_moves = chunk_with_moves.with_columns([
            pl.col("parsed_moves").list.len().alias("num_moves")
        ])
        
        processed_chunks.append(chunk_with_moves)
        
        # Log progress for large datasets
        if total_rows > 5000:
            logger.debug(f"Processed chunk {start_idx//chunk_size + 1}/{(total_rows + chunk_size - 1)//chunk_size}")
    
    # Combine all chunks
    if len(processed_chunks) == 1:
        parsed_df = processed_chunks[0]
    else:
        parsed_df = pl.concat(processed_chunks)
    
    logger.info("Successfully parsed movetext for all games")
    return parsed_df


def main():
    """Main function to process Parquet files."""
    parser = argparse.ArgumentParser(description="Load Parquet files, filter by Elo, and parse movetext")
    parser.add_argument("folder_path", type=str, help="Path to folder containing Parquet files")
    parser.add_argument("--min-elo", type=int, default=1000, help="Minimum Elo rating (default: 1000)")
    parser.add_argument("--output", type=str, help="Output Parquet file path (for single file mode)")
    parser.add_argument("--output-dir", type=str, help="Output directory for batch processing (default: current directory)")
    parser.add_argument("--output-prefix", type=str, default="processed_games", help="Prefix for batch output files (default: processed_games)")
    parser.add_argument("--batch-size", type=int, default=50000, help="Games per batch file (default: 50000)")
    parser.add_argument("--batch-mode", action="store_true", help="Use batch processing for large datasets")
    parser.add_argument("--file-chunk-size", type=int, default=5000, help="Games to read from each file at once (default: 5000)")
    parser.add_argument("--parse-chunk-size", type=int, default=500, help="Games to parse movetext for at once (default: 500)")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads for parallel processing (0 = auto-detect, default: 0)")
    parser.add_argument("--sample", type=int, help="Sample N games for testing (only in single file mode)")
    parser.add_argument("--show-examples", action="store_true", help="Show example parsed moves (only in single file mode)")
    
    args = parser.parse_args()
    
    try:
        folder_path = Path(args.folder_path)
        
        if args.batch_mode:
            # Batch processing mode for large datasets
            logger.info("Using batch processing mode for large dataset")
            
            output_dir = Path(args.output_dir) if args.output_dir else Path(".")
            
            output_files = process_parquet_files_in_batches(
                folder_path=folder_path,
                min_elo=args.min_elo,
                batch_size=args.batch_size,
                output_dir=output_dir,
                output_prefix=args.output_prefix,
                file_chunk_size=args.file_chunk_size,
                parse_chunk_size=args.parse_chunk_size,
                num_threads=args.threads
            )
            
            print(f"\nBatch processing complete!")
            print(f"Created {len(output_files)} batch files in {output_dir}")
            print(f"Batch files: {[f.name for f in output_files[:5]]}...")  # Show first 5 filenames
            
        else:
            # Single file processing mode (original functionality)
            logger.info("Using single file processing mode")
            logger.warning("Note: For datasets >5GB, consider using --batch-mode for better memory efficiency")
            
            # Load Parquet files
            df = load_parquet_files(folder_path)
            
            # Sample if requested (for testing)
            if args.sample:
                df = df.sample(args.sample, seed=42)
                logger.info(f"Sampled {args.sample} games for processing")
            
            # Filter by Elo
            df = filter_by_elo(df, args.min_elo)
            
            if len(df) == 0:
                logger.warning("No games remaining after Elo filtering")
                return
            
            # Parse movetext to moves with configurable chunk size
            chunk_size = args.parse_chunk_size if len(df) > 1000 else min(args.parse_chunk_size, 100)
            df = add_parsed_moves(df, chunk_size=chunk_size)
            
            # Show statistics
            logger.info(f"Final dataset: {len(df)} games")
            
            # Show move count statistics
            move_stats = df.select([
                pl.col("num_moves").mean().alias("avg_moves"),
                pl.col("num_moves").min().alias("min_moves"),
                pl.col("num_moves").max().alias("max_moves"),
                pl.col("num_moves").median().alias("median_moves")
            ]).to_dicts()[0]
            
            logger.info(f"Move statistics: avg={move_stats['avg_moves']:.1f}, "
                        f"min={move_stats['min_moves']}, max={move_stats['max_moves']}, "
                        f"median={move_stats['median_moves']}")
            
            # Show examples if requested
            if args.show_examples:
                print("\n=== Example Parsed Moves ===")
                examples = df.select(["white", "black", "white_elo", "black_elo", "parsed_moves", "num_moves"]).head(3)
                for i, row in enumerate(examples.to_dicts()):
                    print(f"\nGame {i+1}: {row['white']} ({row['white_elo']}) vs {row['black']} ({row['black_elo']})")
                    print(f"Moves ({row['num_moves']}): {row['parsed_moves'][:10]}...")  # Show first 10 moves
            
            # Save output if requested
            if args.output:
                output_path = Path(args.output)
                df.write_parquet(output_path)
                logger.info(f"Saved processed data to {output_path}")
            
            print(f"\nProcessing complete! {len(df)} games processed.")
        
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
