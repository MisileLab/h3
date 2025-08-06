"""Module for parsing very large PGN files line by line."""

import re
import os
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from tqdm import tqdm
import chess  # For converting SAN to UCI

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LargePGNParser:
  """Parser for very large PGN files that processes line by line."""

  def __init__(
    self, 
    batch_size: int = 10000,
    output_dir: Optional[Path] = None,
    min_elo: int = 0,
    output_filename_prefix: str = "chess_games",
    num_workers: int = 0
  ) -> None:
    """Initialize the parser.

    Args:
      batch_size: Number of games to process in a batch.
      output_dir: Directory to save Parquet files. If None, uses current directory.
      min_elo: Minimum Elo rating for games to include.
      output_filename_prefix: Prefix for output Parquet filenames.
      num_workers: Number of worker threads to use for parallel processing. If 0 or negative,
                  uses all available CPU cores.
    """
    self.batch_size = batch_size
    self.output_dir = Path(output_dir) if output_dir else Path(".")
    self.min_elo = min_elo
    self.output_filename_prefix = output_filename_prefix
    
    # Use all available CPU cores if num_workers is 0
    if num_workers <= 0:
      import os
      self.num_workers = os.cpu_count() or 1  # Default to 1 if cpu_count returns None
    else:
      self.num_workers = num_workers
    
    # Create output directory if it doesn't exist
    self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Regex patterns
    self.header_pattern = re.compile(r'\[(.*?)\s+"(.*?)"\]')
    self.move_pattern = re.compile(r'(\d+\.)(\s+\{[^}]*\})?\s*([^ {]+)(\s+\{[^}]*\})?\s*([^ {]+)?(\s+\{[^}]*\})?')
    self.move_only_pattern = re.compile(r'([a-zA-Z0-9+#=\-]+|O O|O O O)')
    self.clock_pattern = re.compile(r'\[%clk\s+(\d+:\d+:\d+)\]')
    self.eval_pattern = re.compile(r'\[%eval\s+([-+]?\d+\.\d+|#[-+]?\d+)\]')
  
  def _extract_headers(self, headers_text: str) -> Dict[str, str]:
    """Extract headers from PGN header text.

    Args:
      headers_text: Text containing PGN headers.

    Returns:
      Dictionary of header keys and values.
    """
    headers = {}
    for match in self.header_pattern.finditer(headers_text):
      key, value = match.groups()
      headers[key] = value
    return headers
  
  def _extract_moves_and_times(self, moves_text: str) -> Tuple[list[str], list[float]]:
    """Extract moves and clock times from PGN moves text.

    Args:
      moves_text: Text containing PGN moves.

    Returns:
      Tuple of (moves, clock_times) with moves in UCI format.
    """
    clock_times = []
    
    # Clean up the moves text - remove result and extra whitespace
    cleaned_text = re.sub(r'(1-0|0-1|1/2-1/2|\*)\s*$', '', moves_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Extract all moves directly using the move_only_pattern
    # This pattern matches only the actual moves like e4, Nf3, O-O, etc.
    move_matches = self.move_only_pattern.findall(cleaned_text)
    
    # Filter out move numbers, result markers, and incorrect chess moves
    san_moves = []  # Store SAN moves for conversion to UCI
    
    # Regex for valid chess moves in UCI format (e.g., e2e4) or standard algebraic notation (e.g., e4, Nf3)
    # Also matches castling notation in both formats (O-O, O-O-O, O O, O O O)
    valid_move_pattern = re.compile(r'^(?:[a-h][1-8][a-h][1-8]|[KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?|O\s+O(?:\s+O)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8])(?:[+#])?$')
    
    # Regex to identify UCI format moves (source square to target square)
    uci_pattern = re.compile(r'^[a-h][1-8][a-h][1-8](?:[qrbn])?$')
    
    for move in move_matches:
        # Process only valid chess moves (not numbers, result markers, etc.) that match our pattern
        if (not move.isdigit() and 
            move not in ['1-0', '0-1', '1/2-1/2', '*', 'clk', 'eval'] and
            not re.match(r'^\d+\.', move) and
            not move.startswith('%') and
            re.match(valid_move_pattern, move)):
            
            # Convert "O O" to "O-O" for consistency
            if move == "O O":
                move = "O-O"
            elif move == "O O O":
                move = "O-O-O"
            
            # Add to SAN moves list for later conversion
            san_moves.append(move)
    
    # Convert SAN moves to UCI format
    valid_moves = []
    for i, move in enumerate(san_moves):
        # Skip conversion if already in UCI format
        if re.match(uci_pattern, move):
            valid_moves.append(move)
        else:
            # Convert SAN to UCI using chess library
            uci_move = self._san_to_uci(move, san_moves[:i])
            valid_moves.append(uci_move)
    
    # Extract clock times
    clock_matches = self.clock_pattern.findall(cleaned_text)
    for time_str in clock_matches:
        h, m, s = map(float, time_str.split(':'))
        clock_times.append(h * 3600 + m * 60 + s)
    
    return valid_moves, clock_times
  
  def _san_to_uci(self, san_move: str, moves_history: List[str]) -> str:
    """Convert a move in Standard Algebraic Notation (SAN) to UCI notation.
    
    Args:
      san_move: Move in SAN format (e.g., "e4", "Nf3").
      moves_history: List of previous moves in SAN format to establish the board position.
      
    Returns:
      Move in UCI format (e.g., "e2e4", "g1f3").
    """
    # Handle castling directly
    castling_kingside = {"O-O", "O O"}
    castling_queenside = {"O-O-O", "O O O"}
    if san_move in castling_kingside:
      return "e1g1" if len(moves_history) % 2 == 0 else "e8g8"
    if san_move in castling_queenside:
      return "e1c1" if len(moves_history) % 2 == 0 else "e8c8"
    
    # Check if the move is already in UCI format
    if re.match(r'^[a-h][1-8][a-h][1-8][qrbnQRBN]?$', san_move):
      return san_move.lower()
    
    # Determine if it's white or black to move
    is_white_move = len(moves_history) % 2 == 0
    
    # Parse pawn moves (e.g., e4, d5)
    if pawn_move := re.match(r'^([a-h])([1-8])$', san_move):
      file = pawn_move.group(1)
      rank = pawn_move.group(2)
      target_rank = int(rank)
      
      # Determine source rank based on whose turn it is and target rank
      if is_white_move:
        # For white pawns
        source_rank = '2' if (target_rank == 4 and not moves_history) or target_rank == 3 else str(target_rank - 1)
      else:
        # For black pawns
        source_rank = '7' if (target_rank == 5 and len(moves_history) == 1) or target_rank == 6 else str(target_rank + 1)
      
      return f"{file}{source_rank}{file}{rank}"
    
    # Parse pawn captures (e.g., exd5, fxg6)
    if pawn_capture := re.match(r'^([a-h])x([a-h])([1-8])$', san_move):
      source_file = pawn_capture.group(1)
      target_file = pawn_capture.group(2)
      target_rank = pawn_capture.group(3)
      
      # Adjust source rank based on target rank
      source_rank = str(int(target_rank) - 1) if is_white_move else str(int(target_rank) + 1)
        
      return f"{source_file}{source_rank}{target_file}{target_rank}"
    
    # Parse piece moves (e.g., Nf3, Bc5)
    piece_move = re.match(r'^([NBRQK])([a-h])([1-8])$', san_move)
    if piece_move:
      piece = piece_move.group(1)
      target_file = piece_move.group(2)
      target_rank = piece_move.group(3)
      
      # Use a more generic approach for piece moves
      if is_white_move:
        if piece == 'N':  # Knights
          if target_file in ['c', 'f'] and target_rank in ['3']:  # Nc3, Nf3
            source_file = 'b' if target_file == 'c' else 'g'
            return f"{source_file}1{target_file}{target_rank}"
        elif piece == 'B':  # Bishops
          if target_file in ['c', 'b', 'f'] and target_rank in ['4', '5']:  # Bc4, Bf5
            source_file = 'f' if target_file in ['c', 'b'] else 'c'
            return f"{source_file}1{target_file}{target_rank}"
      else:
        if piece == 'N':  # Knights
          if target_file in ['c', 'f'] and target_rank in ['6']:  # Nc6, Nf6
            source_file = 'b' if target_file == 'c' else 'g'
            return f"{source_file}8{target_file}{target_rank}"
        elif piece == 'B':  # Bishops
          if target_file in ['c', 'b', 'f'] and target_rank in ['4', '5']:  # Bc5, Bf4
            source_file = 'f' if target_file in ['c', 'b'] else 'c'
            return f"{source_file}8{target_file}{target_rank}"
    
    # For other moves, use chess library to parse
    try:
      board = chess.Board()
      
      # Apply previous moves to get the current position
      for prev_move in moves_history:
        try:
          # Skip castling moves which we'll handle separately
          castling_kingside = {"O-O", "O O"}
          castling_queenside = {"O-O-O", "O O O"}
          if prev_move in castling_kingside:
            move = chess.Move.from_uci("e1g1" if board.turn == chess.WHITE else "e8g8")
          elif prev_move in castling_queenside:
            move = chess.Move.from_uci("e1c1" if board.turn == chess.WHITE else "e8c8")
          else:
            move = board.parse_san(prev_move)
          board.push(move)
        except ValueError:
          # Skip invalid moves
          continue
      
      # Parse the current move
      move = board.parse_san(san_move)
      return move.uci()
    except ValueError:
      # If parsing fails, return the original move
      return san_move
  
  def _extract_time_control(self, time_control: str) -> Tuple[int, int]:
    """Extract base time and increment from time control string.

    Args:
      time_control: Time control string (e.g., "60+0").

    Returns:
      Tuple of (base_time_seconds, increment_seconds).
    """
    if not time_control or '+' not in time_control:
      return 0, 0
    
    parts = time_control.split('+')
    if len(parts) != 2:
      return 0, 0
    
    try:
      base_time = int(parts[0])
      increment = int(parts[1])
      return base_time, increment
    except ValueError:
      return 0, 0
  

  
  def _reader_thread(self, pgn_path: Path, game_queue: queue.Queue, show_progress: bool = True) -> None:
    """Thread that reads the PGN file line by line and puts game content into the queue.

    Args:
      pgn_path: Path to the PGN file.
      game_queue: Queue to put game content into.
      show_progress: Whether to show a progress bar.
    """
    # Get file size for progress bar
    file_size = pgn_path.stat().st_size
    
    # Initialize game parsing state
    current_game_headers = ""
    current_game_moves = ""
    in_headers = False
    in_moves = False
    
    # Open the PGN file
    pbar = None
    if show_progress:
      pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading PGN")
    
    with open(pgn_path, 'r', encoding='utf-8', errors='replace') as f:
      # Process the file line by line
      for line in f:
        # Update progress bar
        if show_progress and pbar is not None:
          pbar.update(len(line))
        
        # Check for start of a new game
        if line.startswith('[Event '):
          # Process the previous game if there is one
          if current_game_headers and current_game_moves:
            # Put the game into the queue
            game_queue.put((current_game_headers, current_game_moves))
          
          # Start a new game
          current_game_headers = line
          current_game_moves = ""
          in_headers = True
          in_moves = False
          continue
        
        # Collect headers
        if in_headers:
          if line.strip():
            current_game_headers += line
          else:
            in_headers = False
            in_moves = True
            continue
        
        # Collect moves
        if in_moves:
          current_game_moves += line
      
      # Process the last game
      if current_game_headers and current_game_moves:
        # Put the last game into the queue
        game_queue.put((current_game_headers, current_game_moves))
    
    # Signal that we're done reading
    game_queue.put(None)
    
    # Close progress bar
    if show_progress and pbar is not None:
      pbar.close()
  
  def _worker_thread(self, worker_id: int, game_queue: queue.Queue, 
                    result_queue: queue.Queue, show_progress: bool = True) -> None:
    """Worker thread that processes games from the queue.

    Args:
      worker_id: ID of the worker thread.
      game_queue: Queue to get game content from.
      result_queue: Queue to put processed games into.
      show_progress: Whether to show a progress bar.
    """
    games_processed = 0
    games_included = 0
    
    pbar = None
    if show_progress:
      pbar = tqdm(desc=f"Worker {worker_id}", position=worker_id + 1, leave=True)
    
    while True:
      # Get a game from the queue
      game_data = game_queue.get()
      
      # Check if we're done
      if game_data is None:
        # Put the None back for other workers to see
        game_queue.put(None)
        break
      
      try:
        # Process the game
        headers_text, moves_text = game_data
        game_record = self._process_game(headers_text, moves_text)
        games_processed += 1
        
        # Apply Elo filter
        white_elo = int(game_record.get('white_elo', 0))
        black_elo = int(game_record.get('black_elo', 0))
        
        if white_elo >= self.min_elo and black_elo >= self.min_elo:
          # Put the processed game into the result queue
          result_queue.put(game_record)
          games_included += 1
        
        # Update progress bar
        if show_progress and pbar is not None:
          pbar.update(1)
      
      except Exception as e:
        logger.error(f"Error processing game in worker {worker_id}: {e}")
    
    # Put the results into the result queue
    result_queue.put((games_processed, games_included))
    
    # Close progress bar
    if show_progress and pbar is not None:
      pbar.close()
  
  def _batch_saver_thread(self, result_queue: queue.Queue, 
                         stop_event: threading.Event, stats: Dict[str, int],
                         show_progress: bool = True) -> None:
    """Thread that saves processed games in batches.

    Args:
      result_queue: Queue to get processed games from.
      stop_event: Event to signal when all workers are done.
      show_progress: Whether to show a progress bar.

    Returns:
      Tuple of (total_games_processed, total_games_included, batch_num).
    """
    batch_num = 0
    current_batch = []
    total_games_processed = 0
    total_games_included = 0
    
    pbar = None
    if show_progress:
      pbar = tqdm(desc="Saving batches", position=0, leave=True)
    
    while not (stop_event.is_set() and result_queue.empty()):
      try:
        # Get a result from the queue with a timeout
        result = result_queue.get(timeout=0.1)
        
        # Check if it's a game record or worker stats
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], int):
          # It's worker stats
          games_processed, games_included = result
          total_games_processed += games_processed
          total_games_included += games_included
          
          # Update progress bar
          if show_progress and pbar is not None:
            pbar.set_postfix({"processed": total_games_processed, "included": total_games_included})
        else:
          # It's a game record
          current_batch.append(result)
          
          # Save batch if it's full
          if len(current_batch) >= self.batch_size:
            self._save_batch(current_batch, batch_num)
            current_batch = []
            batch_num += 1
            
            # Update progress bar
            if show_progress and pbar is not None:
              pbar.update(1)
      
      except queue.Empty:
        # Queue is empty but workers might still be running
        continue
      except Exception as e:
        logger.error(f"Error in batch saver thread: {e}")
    
    # Process the final batch
    if current_batch:
      self._save_batch(current_batch, batch_num)
      batch_num += 1
      
      # Update progress bar
      if show_progress and pbar is not None:
        pbar.update(1)
    
    # Close progress bar
    if show_progress and pbar is not None:
      pbar.close()
    
    # Update the stats dictionary with the final values
    stats["total_games_processed"] = total_games_processed
    stats["total_games_included"] = total_games_included
    stats["batch_num"] = batch_num
  
  def parse_pgn_file(
    self, 
    pgn_path: Union[str, Path],
    show_progress: bool = True
  ) -> int:
    """Parse a large PGN file line by line and save to Parquet.

    Args:
      pgn_path: Path to the PGN file.
      show_progress: Whether to show a progress bar.

    Returns:
      Number of games processed.
    """
    pgn_path = Path(pgn_path)
    logger.info(f"Parsing large PGN file: {pgn_path} with {self.num_workers} workers")
    
    # If single-threaded, use the original implementation
    if self.num_workers == 1:
      return self._parse_pgn_file_single_threaded(pgn_path, show_progress)
    
    # Create queues for communication between threads
    game_queue = queue.Queue(maxsize=self.num_workers * 10)  # Buffer for games to process
    result_queue = queue.Queue()  # Queue for processed games
    
    # Create a stop event to signal when all workers are done
    stop_event = threading.Event()
    
    # Dictionary to store statistics
    stats = {"total_games_processed": 0, "total_games_included": 0, "batch_num": 0}
    
    # Start the reader thread
    logger.info("Starting reader thread...")
    reader_thread = threading.Thread(
      target=self._reader_thread,
      args=(pgn_path, game_queue, show_progress)
    )
    reader_thread.daemon = True
    reader_thread.start()
    
    # Start worker threads
    logger.info(f"Starting {self.num_workers} worker threads...")
    worker_threads = []
    for i in range(self.num_workers):
      worker = threading.Thread(
        target=self._worker_thread,
        args=(i, game_queue, result_queue, show_progress)
      )
      worker.daemon = True
      worker.start()
      worker_threads.append(worker)
    
    # Start the batch saver thread
    logger.info("Starting batch saver thread...")
    batch_saver_thread = threading.Thread(
      target=self._batch_saver_thread,
      args=(result_queue, stop_event, stats, show_progress)
    )
    batch_saver_thread.daemon = True
    batch_saver_thread.start()
    
    # Wait for all worker threads to finish
    for worker in worker_threads:
      worker.join()
    
    # Signal that all workers are done
    stop_event.set()
    
    # Wait for the batch saver thread to finish
    batch_saver_thread.join()
    
    # Get the final stats
    logger.info(f"Processed {stats['total_games_processed']} games, included {stats['total_games_included']} games")
    logger.info(f"Saved {stats['batch_num']} Parquet files to {self.output_dir}")
    
    return stats["total_games_processed"]
  
  def _parse_pgn_file_single_threaded(
    self, 
    pgn_path: Path,
    show_progress: bool = True
  ) -> int:
    """Parse a large PGN file line by line in single-threaded mode.

    Args:
      pgn_path: Path to the PGN file.
      show_progress: Whether to show a progress bar.

    Returns:
      Number of games processed.
    """
    # Get file size for progress bar
    file_size = pgn_path.stat().st_size
    
    # Initialize counters
    games_processed = 0
    games_included = 0
    batch_num = 0
    current_batch = []
    
    # Initialize game parsing state
    current_game_headers = ""
    current_game_moves = ""
    in_headers = False
    in_moves = False
    
    # Open the PGN file
    pbar = None
    with open(pgn_path, 'r', encoding='utf-8', errors='replace') as f:
      # Set up progress bar
      if show_progress:
        pbar = tqdm(total=file_size, unit='B', unit_scale=True)
      
      # Process the file line by line
      for line in f:
        # Update progress bar
        if show_progress and pbar is not None:
          pbar.update(len(line))
        
        # Check for start of a new game
        if line.startswith('[Event '):
          # Process the previous game if there is one
          if current_game_headers and current_game_moves:
            game_record = self._process_game(current_game_headers, current_game_moves)
            games_processed += 1
            
            # Apply Elo filter
            white_elo = int(game_record.get('white_elo', 0))
            black_elo = int(game_record.get('black_elo', 0))
            
            if white_elo >= self.min_elo and black_elo >= self.min_elo:
              current_batch.append(game_record)
              games_included += 1
            
            # Process batch if it's full
            if len(current_batch) >= self.batch_size:
              self._save_batch(current_batch, batch_num)
              current_batch = []
              batch_num += 1
          
          # Start a new game
          current_game_headers = line
          current_game_moves = ""
          in_headers = True
          in_moves = False
          continue
        
        # Collect headers
        if in_headers:
          if line.strip():
            current_game_headers += line
          else:
            in_headers = False
            in_moves = True
            continue
        
        # Collect moves
        if in_moves:
          current_game_moves += line
      
      # Process the last game
      if current_game_headers and current_game_moves:
        game_record = self._process_game(current_game_headers, current_game_moves)
        games_processed += 1
        
        # Apply Elo filter
        white_elo = int(game_record.get('white_elo', 0))
        black_elo = int(game_record.get('black_elo', 0))
        
        if white_elo >= self.min_elo and black_elo >= self.min_elo:
          current_batch.append(game_record)
          games_included += 1
      
      # Process the final batch
      if current_batch:
        self._save_batch(current_batch, batch_num)
      
      # Close progress bar
      if show_progress and pbar is not None:
        pbar.close()
    
    logger.info(f"Processed {games_processed} games, included {games_included} games")
    logger.info(f"Saved {batch_num + 1} Parquet files to {self.output_dir}")
    
    return games_processed
  
  def _process_game(self, headers_text: str, moves_text: str) -> Dict[str, Any]:
    """Process a single game.

    Args:
      headers_text: Text containing PGN headers.
      moves_text: Text containing PGN moves.

    Returns:
      Dictionary with game data.
    """
    # Extract headers
    headers = self._extract_headers(headers_text)
    
    # Extract moves and clock times
    moves, clock_times = self._extract_moves_and_times(moves_text)
    
    # Extract time control
    base_time, increment = self._extract_time_control(headers.get('TimeControl', ''))
    
    # Calculate time usage statistics
    avg_time_per_move = sum(clock_times) / len(clock_times) if clock_times else None
    min_time_per_move = min(clock_times) if clock_times else None
    max_time_per_move = max(clock_times) if clock_times else None
    
    # Create game record
    return {
      'id': headers.get('Site', '').split('/')[-1],
      'event': headers.get('Event', ''),
      'site': headers.get('Site', ''),
      'date': headers.get('UTCDate', headers.get('Date', '')),
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
      'moves': moves,  # Store as list instead of joined string
      'num_moves': len(moves),
      'has_clock_times': bool(clock_times),
      'avg_time_per_move': avg_time_per_move,
      'min_time_per_move': min_time_per_move,
      'max_time_per_move': max_time_per_move,
    }
  
  def _save_batch(self, batch: List[Dict[str, Any]], batch_num: int) -> Path:
    """Save a batch of games to Parquet.

    Args:
      batch: List of game records.
      batch_num: Batch number.

    Returns:
      Path to the saved Parquet file.
    """
    # Convert to Polars DataFrame
    df = pl.DataFrame(batch)
    
    # Save to Parquet
    output_path = self.output_dir / f"{self.output_filename_prefix}_batch_{batch_num:06d}.parquet"
    df.write_parquet(output_path)
    
    logger.info(f"Saved batch {batch_num} with {len(batch)} games to {output_path}")
    
    return output_path
  
  def merge_parquet_files(
    self, 
    output_path: Optional[Path] = None
  ) -> Optional[Path]:
    """Merge all Parquet files into one.

    Args:
      output_path: Path to the output Parquet file. If None, uses default naming.

    Returns:
      Path to the merged Parquet file, or None if no files were found.
    """
    # Find all batch files
    batch_files = sorted(self.output_dir.glob(f"{self.output_filename_prefix}_batch_*.parquet"))
    
    if not batch_files:
      logger.warning("No batch files found to merge")
      return None
    
    logger.info(f"Merging {len(batch_files)} Parquet files")
    
    # Set output path
    if output_path is None:
      output_path = self.output_dir / f"{self.output_filename_prefix}_merged.parquet"
    
    # Read and concatenate all DataFrames
    dfs = []
    for file in tqdm(batch_files, desc="Reading batch files"):
      df = pl.read_parquet(file)
      dfs.append(df)
    
    merged_df = pl.concat(dfs)
    
    # Save the merged DataFrame
    merged_df.write_parquet(output_path)
    
    logger.info(f"Merged {len(batch_files)} files with {len(merged_df)} total rows to {output_path}")
    
    return output_path
