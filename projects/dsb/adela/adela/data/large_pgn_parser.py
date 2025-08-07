"""Optimized module for parsing very large PGN files line by line."""

import re
import os
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import mmap

import polars as pl
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LargePGNParser:
    """Optimized parser for very large PGN files that processes line by line."""

    def __init__(
        self, 
        batch_size: int = 10000,
        output_dir: Optional[Path] = None,
        min_elo: int = 0,
        output_filename_prefix: str = "chess_games",
        num_workers: int = 0,
        buffer_size: int = 1024 * 1024  # 1MB buffer
    ) -> None:
        """Initialize the parser.

        Args:
            batch_size: Number of games to process in a batch.
            output_dir: Directory to save Parquet files. If None, uses current directory.
            min_elo: Minimum Elo rating for games to include.
            output_filename_prefix: Prefix for output Parquet filenames.
            num_workers: Number of worker threads to use for parallel processing.
            buffer_size: Size of file reading buffer in bytes.
        """
        self.batch_size = batch_size
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.min_elo = min_elo
        self.output_filename_prefix = output_filename_prefix
        self.buffer_size = buffer_size
        
        # Use optimal number of workers based on CPU cores
        if num_workers <= 0:
            self.num_workers = min(os.cpu_count() or 1, 8)  # Cap at 8 to avoid overhead
        else:
            self.num_workers = num_workers
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-compiled regex patterns for better performance
        self._compile_patterns()
        
        # Pre-allocated buffers for reuse
        self._game_buffer = []
        
    def _compile_patterns(self) -> None:
        """Pre-compile all regex patterns for better performance."""
        # More efficient header pattern
        self.header_pattern = re.compile(rb'\[(.*?)\s+"(.*?)"\]')
        
        # Optimized move patterns
        self.move_pattern = re.compile(
            rb'(\d+\.)(\s+\{[^}]*\})?\s*([^ {]+)(\s+\{[^}]*\})?\s*([^ {]+)?(\s+\{[^}]*\})?'
        )
        
        # More precise move-only pattern
        self.move_only_pattern = re.compile(
            rb'(?:^|\s)([a-zA-Z0-9+#=\-O]{2,}|O-O(?:-O)?)\s*(?=\s|$|[{\[])'
        )
        
        # Optimized time patterns
        self.clock_pattern = re.compile(rb'\[%clk\s+(\d+:\d+:\d+)\]')
        self.eval_pattern = re.compile(rb'\[%eval\s+([-+]?\d+\.\d+|#[-+]?\d+)\]')
        
        # Common header patterns for fast lookup
        self.common_headers = {
            b'Event', b'Site', b'Date', b'UTCDate', b'UTCTime', b'White', b'Black',
            b'WhiteElo', b'BlackElo', b'Result', b'TimeControl', b'Termination',
            b'ECO', b'Opening', b'WhiteRatingDiff', b'BlackRatingDiff'
        }

    def _extract_headers_optimized(self, headers_bytes: bytes) -> Dict[str, str]:
        """Extract headers from PGN header bytes with optimized processing.

        Args:
            headers_bytes: Bytes containing PGN headers.

        Returns:
            Dictionary of header keys and values.
        """
        headers = {}
        
        # Use compiled regex with bytes
        for match in self.header_pattern.finditer(headers_bytes):
            key_bytes, value_bytes = match.groups()
            
            # Only process headers we care about to save time
            if key_bytes in self.common_headers:
                key = key_bytes.decode('utf-8', errors='ignore')
                value = value_bytes.decode('utf-8', errors='ignore')
                headers[key] = value
        
        return headers

    def _extract_moves_optimized(self, moves_bytes: bytes) -> Tuple[List[str], List[float]]:
        """Extract moves and clock times with optimized processing.

        Args:
            moves_bytes: Bytes containing PGN moves.

        Returns:
            Tuple of (moves, clock_times).
        """
        # Remove result markers and normalize whitespace
        cleaned_bytes = re.sub(rb'(1-0|0-1|1/2-1/2|\*)\s*$', b'', moves_bytes)
        cleaned_bytes = re.sub(rb'\s+', b' ', cleaned_bytes).strip()
        
        # Extract moves more efficiently
        moves = []
        move_matches = self.move_only_pattern.findall(cleaned_bytes)
        
        # Pre-compiled patterns for validation
        valid_move_bytes = re.compile(
            rb'^(?:[a-h][1-8][a-h][1-8]|[KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8])(?:[+#])?$'
        )
        
        for move_bytes in move_matches:
            if (not move_bytes.isdigit() and 
                move_bytes not in {b'1-0', b'0-1', b'1/2-1/2', b'*', b'clk', b'eval'} and
                not re.match(rb'^\d+\.', move_bytes) and
                valid_move_bytes.match(move_bytes)):
                
                move = move_bytes.decode('utf-8', errors='ignore')
                moves.append(move)
        
        # Extract clock times
        clock_times = []
        for time_bytes in self.clock_pattern.findall(cleaned_bytes):
            time_str = time_bytes.decode('utf-8', errors='ignore')
            try:
                h, m, s = map(float, time_str.split(':'))
                clock_times.append(h * 3600 + m * 60 + s)
            except ValueError:
                continue
        
        return moves, clock_times

    def _extract_time_control_fast(self, time_control: str) -> Tuple[int, int]:
        """Fast time control extraction with caching."""
        if not time_control or '+' not in time_control:
            return 0, 0
        
        # Use string splitting instead of regex for simple cases
        parts = time_control.split('+', 1)
        if len(parts) != 2:
            return 0, 0
        
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return 0, 0

    def _read_file_chunked(self, pgn_path: Path) -> Generator[bytes, None, None]:
        """Read file in chunks using memory mapping for better I/O performance."""
        try:
            with open(pgn_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    chunk_start = 0
                    while chunk_start < len(mm):
                        chunk_end = min(chunk_start + self.buffer_size, len(mm))
                        
                        # Find the end of the last complete game in this chunk
                        if chunk_end < len(mm):
                            # Look for the start of the next game
                            next_game_start = mm.find(b'[Event ', chunk_start, chunk_end)
                            if next_game_start != -1 and next_game_start != chunk_start:
                                chunk_end = next_game_start
                        
                        yield mm[chunk_start:chunk_end]
                        chunk_start = chunk_end
                        
        except (OSError, ValueError):
            # Fallback to regular file reading if mmap fails
            with open(pgn_path, 'rb', buffering=self.buffer_size) as f:
                while True:
                    chunk = f.read(self.buffer_size)
                    if not chunk:
                        break
                    yield chunk

    def _reader_thread_optimized(self, pgn_path: Path, game_queue: queue.Queue, 
                               show_progress: bool = True) -> None:
        """Optimized reader thread using memory mapping and byte processing."""
        try:
            file_size = pgn_path.stat().st_size
        except (OSError, FileNotFoundError):
            file_size = None
            logger.warning(f"Could not determine file size for {pgn_path}")
        
        bytes_read = 0
        
        pbar = None
        if show_progress:
            pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading PGN")
        
        # State for parsing
        current_game_data = bytearray()
        in_game = False
        games_sent = 0
        
        try:
            for chunk in self._read_file_chunked(pgn_path):
                bytes_read += len(chunk)
                if pbar:
                    pbar.update(len(chunk))
                
                # Process chunk line by line
                lines = chunk.split(b'\n')
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check for game start
                    if line.startswith(b'[Event '):
                        # Send previous game if exists
                        if in_game and current_game_data:
                            game_queue.put(bytes(current_game_data))
                            games_sent += 1
                            if games_sent % 1000 == 0:  # Periodic logging
                                logger.debug(f"Reader sent {games_sent} games")
                        
                        # Start new game
                        current_game_data = bytearray(line + b'\n')
                        in_game = True
                    elif in_game:
                        current_game_data.extend(line + b'\n')
            
            # Send final game
            if in_game and current_game_data:
                game_queue.put(bytes(current_game_data))
                games_sent += 1
                
        except Exception as e:
            logger.error(f"Error in reader thread: {e}")
        finally:
            # Signal completion
            game_queue.put(None)
            if pbar:
                pbar.close()
            logger.info(f"Reader thread sent {games_sent} games")

    def _worker_thread_optimized(self, worker_id: int, game_queue: queue.Queue, 
                               result_queue: queue.Queue, show_progress: bool = True) -> None:
        """Optimized worker thread with better memory management."""
        games_processed = 0
        games_included = 0
        
        pbar = None
        if show_progress:
            pbar = tqdm(desc=f"Worker {worker_id}", position=worker_id + 1, leave=True)
        
        # Pre-allocate reusable objects
        local_buffer = []
        
        try:
            while True:
                try:
                    game_data = game_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if game_data is None:
                    # Put the None back for other workers
                    game_queue.put(None)
                    break
                
                try:
                    # Process game from bytes
                    game_record = self._process_game_bytes(game_data)
                    games_processed += 1
                    
                    # Fast Elo filtering
                    white_elo = game_record.get('white_elo', 0)
                    black_elo = game_record.get('black_elo', 0)
                    
                    if white_elo >= self.min_elo and black_elo >= self.min_elo:
                        result_queue.put(game_record)
                        games_included += 1
                    
                    if pbar and games_processed % 100 == 0:  # Update every 100 games
                        pbar.update(100)
                        
                except Exception as e:
                    logger.error(f"Worker {worker_id} error processing game: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Worker {worker_id} fatal error: {e}")
        finally:
            # Send final stats
            result_queue.put((games_processed, games_included))
            if pbar:
                pbar.close()
            logger.info(f"Worker {worker_id} processed {games_processed} games, included {games_included}")

    def _process_game_bytes(self, game_bytes: bytes) -> Dict[str, Any]:
        """Process a single game from bytes with optimized parsing."""
        # Split headers and moves more efficiently
        game_text = game_bytes.decode('utf-8', errors='ignore')
        lines = game_text.strip().split('\n')
        
        headers_lines = []
        moves_lines = []
        in_headers = True
        
        for line in lines:
            if line.startswith('[') and in_headers:
                headers_lines.append(line)
            elif line.strip() == '' and in_headers:
                in_headers = False
            elif not in_headers:
                moves_lines.append(line)
        
        headers_text = '\n'.join(headers_lines)
        moves_text = '\n'.join(moves_lines)
        
        # Use optimized extraction methods
        headers = self._extract_headers_optimized(headers_text.encode('utf-8'))
        moves, clock_times = self._extract_moves_optimized(moves_text.encode('utf-8'))
        
        # Fast time control processing
        base_time, increment = self._extract_time_control_fast(headers.get('TimeControl', ''))
        
        # Optimized statistics calculation
        time_stats = self._calculate_time_stats(clock_times)
        
        # Create optimized game record with type hints for better performance
        return {
            'id': headers.get('Site', '').split('/')[-1] if headers.get('Site') else '',
            'event': headers.get('Event', ''),
            'site': headers.get('Site', ''),
            'date': headers.get('UTCDate') or headers.get('Date', ''),
            'time': headers.get('UTCTime', ''),
            'white': headers.get('White', ''),
            'black': headers.get('Black', ''),
            'white_elo': int(headers.get('WhiteElo', '0')),
            'black_elo': int(headers.get('BlackElo', '0')),
            'white_rating_diff': int(headers.get('WhiteRatingDiff', '0')),
            'black_rating_diff': int(headers.get('BlackRatingDiff', '0')),
            'result': headers.get('Result', '*'),
            'time_control': headers.get('TimeControl', ''),
            'base_time_seconds': base_time,
            'increment_seconds': increment,
            'termination': headers.get('Termination', ''),
            'eco': headers.get('ECO', ''),
            'opening': headers.get('Opening', ''),
            'moves': moves,
            'num_moves': len(moves),
            'has_clock_times': bool(clock_times) if clock_times is not None else False,
            **time_stats
        }

    def _calculate_time_stats(self, clock_times: List[float]) -> Dict[str, Optional[float]]:
        """Optimized time statistics calculation."""
        if not clock_times:
            return {
                'avg_time_per_move': None,
                'min_time_per_move': None,
                'max_time_per_move': None
            }
        
        # Use built-in functions for better performance
        total_time = sum(clock_times)
        return {
            'avg_time_per_move': total_time / len(clock_times),
            'min_time_per_move': min(clock_times),
            'max_time_per_move': max(clock_times)
        }

    def _batch_saver_thread_optimized(self, result_queue: queue.Queue, 
                                    stop_event: threading.Event, stats: Dict[str, int],
                                    show_progress: bool = True) -> None:
        """Optimized batch saver with better memory management."""
        batch_num = 0
        current_batch = []
        total_games_processed = 0
        total_games_included = 0
        
        pbar = None
        if show_progress:
            pbar = tqdm(desc="Saving batches", position=0, leave=True)
        
        # Pre-allocate batch list with expected size
        # Note: Python lists don't have a reserve method, this is just a comment
        
        try:
            while not (stop_event.is_set() and result_queue.empty()):
                try:
                    result = result_queue.get(timeout=0.1)
                    
                    if isinstance(result, tuple) and len(result) == 2:
                        # Worker stats
                        games_processed, games_included = result
                        total_games_processed += games_processed
                        total_games_included += games_included
                        
                        if pbar:
                            pbar.set_postfix({
                                "processed": total_games_processed, 
                                "included": total_games_included
                            })
                    else:
                        # Game record
                        current_batch.append(result)
                        
                        if len(current_batch) >= self.batch_size:
                            self._save_batch_optimized(current_batch, batch_num)
                            current_batch.clear()  # More efficient than creating new list
                            batch_num += 1
                            
                            if pbar:
                                pbar.update(1)
                                
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in batch saver: {e}")
            
            # Save final batch
            if current_batch:
                self._save_batch_optimized(current_batch, batch_num)
                batch_num += 1
                if pbar:
                    pbar.update(1)
                    
        finally:
            if pbar:
                pbar.close()
            
            # Update stats
            stats.update({
                "total_games_processed": total_games_processed,
                "total_games_included": total_games_included,
                "batch_num": batch_num
            })

    def _save_batch_optimized(self, batch: List[Dict[str, Any]], batch_num: int) -> Optional[Path]:
        """Optimized batch saving with better compression and memory usage."""
        if not batch:
            return None
        
        try:
            # Create DataFrame more efficiently
            df = pl.DataFrame(batch, infer_schema_length=min(len(batch), 1000))
            
            # Save with optimized compression
            output_path = self.output_dir / f"{self.output_filename_prefix}_batch_{batch_num:06d}.parquet"
            df.write_parquet(
                output_path,
                compression='zstd',  # Better compression than default
                use_pyarrow=True,    # Use PyArrow for better performance
            )
            
            logger.debug(f"Saved batch {batch_num} with {len(batch)} games to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving batch {batch_num}: {e}")
            raise

    def parse_pgn_file(
        self, 
        pgn_path: Union[str, Path],
        show_progress: bool = True
    ) -> int:
        """Parse a large PGN file with optimized multi-threading."""
        pgn_path = Path(pgn_path)
        logger.info(f"Parsing PGN file: {pgn_path} with {self.num_workers} workers")
        
        if self.num_workers == 1:
            return self._parse_pgn_file_single_threaded_optimized(pgn_path, show_progress)
        
        # Optimized queue sizes based on number of workers
        game_queue_size = max(self.num_workers * 20, 1000)
        result_queue_size = max(self.num_workers * 10, 500)
        
        game_queue = queue.Queue(maxsize=game_queue_size)
        result_queue = queue.Queue(maxsize=result_queue_size)
        stop_event = threading.Event()
        stats = {"total_games_processed": 0, "total_games_included": 0, "batch_num": 0}
        
        # Start optimized threads
        threads = []
        
        # Reader thread
        reader_thread = threading.Thread(
            target=self._reader_thread_optimized,
            args=(pgn_path, game_queue, show_progress),
            daemon=True
        )
        reader_thread.start()
        threads.append(reader_thread)
        
        # Worker threads
        worker_threads = []
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_thread_optimized,
                args=(i, game_queue, result_queue, show_progress),
                daemon=True
            )
            worker.start()
            worker_threads.append(worker)
            threads.append(worker)
        
        # Batch saver thread
        saver_thread = threading.Thread(
            target=self._batch_saver_thread_optimized,
            args=(result_queue, stop_event, stats, show_progress),
            daemon=True
        )
        saver_thread.start()
        threads.append(saver_thread)
        
        # Wait for completion
        try:
            reader_thread.join()
            for worker in worker_threads:
                worker.join()
            stop_event.set()
            saver_thread.join()
            
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
            stop_event.set()
            
        logger.info(f"Processed {stats['total_games_processed']} games, "
                   f"included {stats['total_games_included']} games")
        logger.info(f"Saved {stats['batch_num']} Parquet files to {self.output_dir}")
        
        return stats["total_games_processed"]

    def _parse_pgn_file_single_threaded_optimized(
        self, 
        pgn_path: Path,
        show_progress: bool = True
    ) -> int:
        """Optimized single-threaded parsing with memory mapping."""
        try:
            file_size = pgn_path.stat().st_size
        except (OSError, FileNotFoundError):
            file_size = None
            logger.warning(f"Could not determine file size for {pgn_path}")
        
        games_processed = 0
        games_included = 0
        batch_num = 0
        current_batch = []
        
        pbar = None
        if show_progress:
            pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Processing PGN")
        
        try:
            bytes_processed = 0
            current_game = bytearray()
            in_game = False
            
            for chunk in self._read_file_chunked(pgn_path):
                bytes_processed += len(chunk)
                if pbar:
                    pbar.update(len(chunk))
                
                lines = chunk.split(b'\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith(b'[Event '):
                        # Process previous game
                        if in_game and current_game:
                            try:
                                game_record = self._process_game_bytes(bytes(current_game))
                                games_processed += 1
                                
                                white_elo = game_record.get('white_elo', 0)
                                black_elo = game_record.get('black_elo', 0)
                                
                                if white_elo >= self.min_elo and black_elo >= self.min_elo:
                                    current_batch.append(game_record)
                                    games_included += 1
                                
                                if len(current_batch) >= self.batch_size:
                                    self._save_batch_optimized(current_batch, batch_num)
                                    current_batch.clear()
                                    batch_num += 1
                                    
                            except Exception as e:
                                logger.error(f"Error processing game {games_processed}: {e}")
                        
                        # Start new game
                        current_game = bytearray(line + b'\n')
                        in_game = True
                    elif in_game:
                        current_game.extend(line + b'\n')
            
            # Process final game
            if in_game and current_game:
                try:
                    game_record = self._process_game_bytes(bytes(current_game))
                    games_processed += 1
                    
                    white_elo = game_record.get('white_elo', 0)
                    black_elo = game_record.get('black_elo', 0)
                    
                    if white_elo >= self.min_elo and black_elo >= self.min_elo:
                        current_batch.append(game_record)
                        games_included += 1
                        
                except Exception as e:
                    logger.error(f"Error processing final game: {e}")
            
            # Save final batch
            if current_batch:
                self._save_batch_optimized(current_batch, batch_num)
                batch_num += 1
                
        finally:
            if pbar:
                pbar.close()
        
        logger.info(f"Single-threaded: Processed {games_processed} games, "
                   f"included {games_included} games")
        logger.info(f"Saved {batch_num} Parquet files to {self.output_dir}")
        
        return games_processed

    def merge_parquet_files_optimized(
        self, 
        output_path: Optional[Path] = None,
        chunk_size: int = 100000
    ) -> Optional[Path]:
        """Optimized Parquet file merging with chunked processing."""
        batch_files = sorted(self.output_dir.glob(f"{self.output_filename_prefix}_batch_*.parquet"))
        
        if not batch_files:
            logger.warning("No batch files found to merge")
            return None
        
        logger.info(f"Merging {len(batch_files)} Parquet files")
        
        if output_path is None:
            output_path = self.output_dir / f"{self.output_filename_prefix}_merged.parquet"
        
        try:
            # Use Polars scan for lazy evaluation
            lazy_dfs = [pl.scan_parquet(file) for file in batch_files]
            merged_lazy = pl.concat(lazy_dfs)
            
            # Write with optimized settings
            merged_lazy.collect().write_parquet(
                output_path,
                compression='zstd',
                use_pyarrow=True,
                row_group_size=chunk_size
            )
            
            # Get final row count efficiently
            row_count = pl.scan_parquet(output_path).select(pl.count()).collect().item()
            
            logger.info(f"Merged {len(batch_files)} files with {row_count} total rows to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error merging Parquet files: {e}")
            return None


# Example usage function
def main():
    """Example usage of the optimized parser."""
    parser = LargePGNParser(
        batch_size=50000,  # Larger batches for better I/O efficiency
        num_workers=0,     # Auto-detect optimal worker count
        min_elo=1200,      # Filter low-rated games
        buffer_size=2 * 1024 * 1024  # 2MB buffer for large files
    )
    
    # Parse the file
    total_games = parser.parse_pgn_file("large_chess_games.pgn", show_progress=True)
    
    # Merge results
    merged_file = parser.merge_parquet_files_optimized()
    
    print(f"Processing complete. Parsed {total_games} games.")
    print(f"Merged file saved to: {merged_file}")


if __name__ == "__main__":
    main()