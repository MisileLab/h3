#!/usr/bin/env python3
"""Script to load Parquet files, filter by Elo, and parse movetext to chess moves."""

import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import gc

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
    
    # Pattern to match chess moves in SAN notation
    # This includes:
    # - Regular moves: e4, Nf3, Bd5, etc.
    # - Captures: exd5, Nxf7, etc.
    # - Castling: O-O, O-O-O
    # - Promotions: e8=Q, a1=R+, etc.
    # - Check/Checkmate: Qg5+, Qe8#, etc.
    move_pattern = r'(?:^|\s)([a-h][1-8]|[KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8])([+#]?)(?=\s|$|[{\[])'
    
    # Find all move matches
    for match in re.finditer(move_pattern, cleaned):
        move = match.group(1) + match.group(2)  # Combine move and check/checkmate marker
        
        # Filter out obvious non-moves
        if (not move.isdigit() and 
            move not in {'1-0', '0-1', '1/2-1/2', '*', 'clk', 'eval'} and
            not re.match(r'^\d+\.', move) and
            len(move) >= 2):
            moves.append(move)
    
    # Alternative approach using split and regex validation if the above doesn't capture all moves
    if not moves:
        # Split on move numbers and extract moves
        tokens = re.split(r'\d+\.', cleaned)
        for token in tokens:
            if not token.strip():
                continue
            
            # Extract individual moves from each token
            token_moves = re.findall(r'[a-h][1-8]|[KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?[+#]?', token)
            
            moves.extend(move for move in token_moves 
                        if move and move not in {'1-0', '0-1', '1/2-1/2', '*'} and len(move) >= 2)
    
    return moves


def process_parquet_files_in_batches(
    folder_path: Path, 
    min_elo: int = 1500, 
    batch_size: int = 50000,
    output_dir: Optional[Path] = None,
    output_prefix: str = "processed_games"
) -> List[Path]:
    """Process Parquet files in batches to handle large datasets efficiently.
    
    Args:
        folder_path: Path to folder containing Parquet files
        min_elo: Minimum Elo rating threshold
        batch_size: Number of games per output batch (default: 50000)
        output_dir: Directory to save processed batches (default: current directory)
        output_prefix: Prefix for output filenames
        
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
    
    # Initialize batch tracking
    current_batch = []
    batch_num = 0
    total_processed = 0
    total_included = 0
    output_files = []
    
    # Process each Parquet file
    for file_idx, parquet_file in enumerate(tqdm(parquet_files, desc="Processing files")):
        logger.info(f"Processing file {file_idx + 1}/{len(parquet_files)}: {parquet_file.name}")
        
        try:
            # Load file with lazy evaluation for memory efficiency
            lazy_df = pl.scan_parquet(str(parquet_file))
            
            # Apply Elo filter at scan level for efficiency
            filtered_lazy = lazy_df.filter(
                (pl.col("white_elo") >= min_elo) | (pl.col("black_elo") >= min_elo)
            )
            
            # Collect in chunks to manage memory
            chunk_size = 10000
            offset = 0
            
            while True:
                # Get chunk
                chunk_df = filtered_lazy.slice(offset, chunk_size).collect()
                
                if len(chunk_df) == 0:
                    break
                
                total_processed += len(chunk_df)
                
                # Parse movetext for this chunk
                chunk_with_moves = add_parsed_moves(chunk_df)
                total_included += len(chunk_with_moves)
                
                # Convert to list of dictionaries for batch accumulation
                chunk_dicts = chunk_with_moves.to_dicts()
                current_batch.extend(chunk_dicts)
                
                # Save batch if it reaches the target size
                if len(current_batch) >= batch_size:
                    output_file = _save_batch(current_batch[:batch_size], batch_num, output_dir, output_prefix)
                    output_files.append(output_file)
                    
                    # Keep remaining items for next batch
                    current_batch = current_batch[batch_size:]
                    batch_num += 1
                    
                    logger.info(f"Saved batch {batch_num} with {batch_size} games")
                    
                    # Force garbage collection to free memory
                    gc.collect()
                
                offset += chunk_size
                
        except Exception as e:
            logger.error(f"Error processing file {parquet_file}: {e}")
            continue
        
        # Clean up memory after each file
        gc.collect()
    
    # Save final batch if it has any games
    if current_batch:
        output_file = _save_batch(current_batch, batch_num, output_dir, output_prefix)
        output_files.append(output_file)
        logger.info(f"Saved final batch {batch_num + 1} with {len(current_batch)} games")
    
    logger.info(f"Processing complete: {total_processed} games processed, {total_included} games included")
    logger.info(f"Created {len(output_files)} batch files in {output_dir}")
    
    return output_files


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
    
    # Generate output filename
    output_file = output_dir / f"{prefix}_batch_{batch_num:06d}.parquet"
    
    # Save with compression
    batch_df.write_parquet(
        output_file,
        compression="zstd",
        use_pyarrow=True
    )
    
    return output_file


def load_parquet_files(folder_path: Path) -> pl.DataFrame:
    """Load all Parquet files from a folder and combine them.
    
    NOTE: This function loads all data into memory and is not suitable for very large datasets.
    Use process_parquet_files_in_batches() for large datasets instead.
    
    Args:
        folder_path: Path to folder containing Parquet files
        
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
    
    # Load and combine all Parquet files
    if len(parquet_files) == 1:
        df = pl.read_parquet(parquet_files[0])
    else:
        # Use scan_parquet for efficient lazy loading and concatenation
        lazy_dfs = [pl.scan_parquet(str(file)) for file in parquet_files]
        df = pl.concat(lazy_dfs).collect()
    
    logger.info(f"Loaded {len(df)} total games from Parquet files")
    return df


def filter_by_elo(df: pl.DataFrame, min_elo: int = 1500) -> pl.DataFrame:
    """Filter games where either whiteElo >= min_elo or blackElo >= min_elo.
    
    Args:
        df: DataFrame containing chess games
        min_elo: Minimum Elo rating threshold
        
    Returns:
        Filtered DataFrame
    """
    original_count = len(df)
    
    # Filter where either white or black player has Elo >= min_elo
    filtered_df = df.filter(
        (pl.col("white_elo") >= min_elo) | (pl.col("black_elo") >= min_elo)
    )
    
    filtered_count = len(filtered_df)
    logger.info(f"Filtered from {original_count} to {filtered_count} games with Elo >= {min_elo}")
    
    return filtered_df


def add_parsed_moves(df: pl.DataFrame) -> pl.DataFrame:
    """Add a column with parsed chess moves from movetext.
    
    Args:
        df: DataFrame containing chess games with movetext column
        
    Returns:
        DataFrame with additional 'parsed_moves' column
    """
    logger.info("Parsing movetext to extract chess moves...")
    
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
    
    # Apply the parsing function to each movetext
    parsed_df = df.with_columns([
        pl.col(movetext_col).map_elements(
            parse_movetext_to_moves, 
            return_dtype=pl.List(pl.Utf8)
        ).alias("parsed_moves")
    ])
    
    # Add some statistics
    parsed_df = parsed_df.with_columns([
        pl.col("parsed_moves").list.len().alias("num_moves")
    ])
    
    logger.info("Successfully parsed movetext for all games")
    return parsed_df


def main():
    """Main function to process Parquet files."""
    parser = argparse.ArgumentParser(description="Load Parquet files, filter by Elo, and parse movetext")
    parser.add_argument("folder_path", type=str, help="Path to folder containing Parquet files")
    parser.add_argument("--min-elo", type=int, default=1500, help="Minimum Elo rating (default: 1500)")
    parser.add_argument("--output", type=str, help="Output Parquet file path (for single file mode)")
    parser.add_argument("--output-dir", type=str, help="Output directory for batch processing (default: current directory)")
    parser.add_argument("--output-prefix", type=str, default="processed_games", help="Prefix for batch output files (default: processed_games)")
    parser.add_argument("--batch-size", type=int, default=50000, help="Games per batch file (default: 50000)")
    parser.add_argument("--batch-mode", action="store_true", help="Use batch processing for large datasets")
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
                output_prefix=args.output_prefix
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
            
            # Parse movetext to moves
            df = add_parsed_moves(df)
            
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
