"""Chess board representation and utilities."""

from typing import final

import bulletchess as chess
import numpy as np


@final
class BoardRepresentation:
  """Chess board representation with utility methods for neural network input."""

  # Channel mapping for one-hot encoding
  # Uppercase for white, lowercase for black handled by offset
  _PIECE_TO_CHANNEL = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
  }

  def __init__(self, fen: str | None = None) -> None:
    """Initialize board representation.

    Args:
      fen: Optional FEN string to initialize the board. If None, uses starting position.
    """
    self.board = chess.Board.from_fen(fen) if fen else chess.Board()

  def get_legal_moves(self):
    """Get all legal moves for the current position.

    Returns:
      List of legal chess moves.
    """
    return self.board.legal_moves()

  def make_move(self, move: chess.Move) -> None:
    """Make a move on the board.

    Args:
      move: Chess move to make.
    """
    self.board.apply(move)

  def undo_move(self) -> None:
    """Undo the last move."""
    _ = self.board.undo()

  def is_game_over(self) -> bool:
    """Check if the game is over.

    Returns:
      True if the game is over, False otherwise.
    """
    return len(self.board.legal_moves()) == 0

  def get_result(self) -> str:
    """Get the game result.

    Returns:
      Game result as a string: "1-0", "0-1", "1/2-1/2", or "*" if the game is not over.
    """
    if not self.is_game_over():
      return "*"
    if self.board in chess.DRAW:
      return "1/2-1/2"
    if self.board in chess.CHECKMATE:
      return "1-0" if self._side_to_move_from_fen() == "b" else "0-1"
    return "*"

  def get_fen(self) -> str:
    """Get the FEN string for the current position.

    Returns:
      FEN string representation.
    """
    return self.board.fen()

  def get_phase(self) -> float:
    """Calculate the game phase.

    Returns:
      A float between 0 and 1, where 0 is the opening and 1 is the endgame.
    """
    # Count material (excluding kings) based on FEN
    board_part = self.get_fen().split()[0]
    total_material = 0
    for ch in board_part:
      if ch in "PNBRQpnbrq":
        if ch.upper() != "K":
          total_material += 1

    # Maximum material (excluding kings) is 30 pieces
    # 8 pawns + 2 knights + 2 bishops + 2 rooks + 1 queen per side
    max_material = 30
    
    # Normalize to 0-1 range and invert (more material = earlier phase)
    phase = 1 - (total_material / max_material)
    return phase

  def get_board_tensor(self) -> np.ndarray:
    """Convert the board to a tensor representation for neural network input.

    Returns:
      A numpy array of shape (12, 8, 8) representing the board.
      Channels are [WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK].
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    board_part = self.get_fen().split()[0]
    rank_idx = 0
    file_idx = 0
    for ch in board_part:
      if ch == '/':
        rank_idx += 1
        file_idx = 0
        continue
      if ch.isdigit():
        file_idx += int(ch)
        continue
      # Piece
      is_white = ch.isupper()
      key = ch.upper()
      channel = self._PIECE_TO_CHANNEL.get(key)
      if channel is None:
        file_idx += 1
        continue
      if not is_white:
        channel += 6
      if 0 <= rank_idx < 8 and 0 <= file_idx < 8:
        tensor[channel, rank_idx, file_idx] = 1.0
      file_idx += 1
    return tensor

  def get_additional_features(self) -> np.ndarray:
    """Get additional features about the position.

    Returns:
      A numpy array with additional features:
      - Side to move (1 for white, 0 for black)
      - Castling rights (4 values)
      - En passant possibility (1 value)
      - Halfmove clock normalized (1 value)
      - Game phase (1 value)
    """
    features = np.zeros(8, dtype=np.float32)
    
    # Side to move
    features[0] = 1.0 if self._side_to_move_from_fen() == "w" else 0.0
    
    fen = self.get_fen()
    try:
      parts = fen.split()
      rights = parts[2] if len(parts) > 2 else "-"
      ep = parts[3] if len(parts) > 3 else "-"
      halfmove = int(parts[4]) if len(parts) > 4 else 0
    except Exception:
      rights = "-"
      ep = "-"
      halfmove = 0

    features[1] = 1.0 if "K" in rights else 0.0
    features[2] = 1.0 if "Q" in rights else 0.0
    features[3] = 1.0 if "k" in rights else 0.0
    features[4] = 1.0 if "q" in rights else 0.0
    features[5] = 0.0 if ep == "-" else 1.0
    features[6] = min(1.0, halfmove / 100.0)
    
    # Game phase
    features[7] = self.get_phase()
    
    return features

  def _move_to_uci(self, move: chess.Move) -> str:
    """Convert a move object to a UCI string representation."""
    return move.uci()

  def get_move_index(self, move: chess.Move) -> int:
    """Convert a move to an index in the policy output.

    Args:
      move: Chess move.

    Returns:
      Integer index for the move.
    """
    return self.get_legal_moves().index(move)

  def get_move_from_index(self, index: int) -> object:
    """Convert a policy index to a chess move.

    Args:
      index: Index in the policy output.

    Returns:
      Chess move.
    """
    return self.get_legal_moves()[index]

  def _side_to_move_from_fen(self) -> str:
    return self.get_fen().split()[1].lower()
