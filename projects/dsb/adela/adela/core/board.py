"""Chess board representation and utilities."""

import chess
import numpy as np
from typing import List, Tuple, Optional, Dict


class BoardRepresentation:
    """Chess board representation with utility methods for neural network input."""

    # Piece type mapping for one-hot encoding
    PIECE_MAPPING = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    def __init__(self, fen: Optional[str] = None) -> None:
        """Initialize board representation.

        Args:
            fen: Optional FEN string to initialize the board. If None, uses starting position.
        """
        self.board = chess.Board(fen) if fen else chess.Board()

    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves for the current position.

        Returns:
            List of legal chess moves.
        """
        return list(self.board.legal_moves)

    def make_move(self, move: chess.Move) -> None:
        """Make a move on the board.

        Args:
            move: Chess move to make.
        """
        self.board.push(move)

    def undo_move(self) -> None:
        """Undo the last move."""
        self.board.pop()

    def is_game_over(self) -> bool:
        """Check if the game is over.

        Returns:
            True if the game is over, False otherwise.
        """
        return self.board.is_game_over()

    def get_result(self) -> str:
        """Get the game result.

        Returns:
            Game result as a string: "1-0", "0-1", "1/2-1/2", or "*" if the game is not over.
        """
        if not self.is_game_over():
            return "*"
        return self.board.result()

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
        # Count material to determine game phase
        total_material = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            total_material += len(self.board.pieces(piece_type, chess.WHITE))
            total_material += len(self.board.pieces(piece_type, chess.BLACK))

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
        
        # Fill tensor with piece positions
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                              chess.ROOK, chess.QUEEN, chess.KING]:
                # Get all squares with this piece type and color
                pieces = self.board.pieces(piece_type, color)
                
                # Calculate channel index
                channel_idx = self.PIECE_MAPPING[piece_type]
                if color == chess.BLACK:
                    channel_idx += 6
                
                # Set corresponding positions in tensor
                for square in pieces:
                    rank, file = chess.square_rank(square), chess.square_file(square)
                    tensor[channel_idx, rank, file] = 1.0
        
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
        features[0] = 1.0 if self.board.turn == chess.WHITE else 0.0
        
        # Castling rights
        features[1] = 1.0 if self.board.has_kingside_castling_rights(chess.WHITE) else 0.0
        features[2] = 1.0 if self.board.has_queenside_castling_rights(chess.WHITE) else 0.0
        features[3] = 1.0 if self.board.has_kingside_castling_rights(chess.BLACK) else 0.0
        features[4] = 1.0 if self.board.has_queenside_castling_rights(chess.BLACK) else 0.0
        
        # En passant
        features[5] = 1.0 if self.board.ep_square is not None else 0.0
        
        # Halfmove clock (normalized to 0-1)
        features[6] = min(1.0, self.board.halfmove_clock / 100.0)
        
        # Game phase
        features[7] = self.get_phase()
        
        return features

    def get_move_index(self, move: chess.Move) -> int:
        """Convert a move to an index in the policy output.

        Args:
            move: Chess move.

        Returns:
            Integer index for the move.
        """
        # Create a mapping of all possible moves
        legal_moves = list(self.board.legal_moves)
        return legal_moves.index(move)

    def get_move_from_index(self, index: int) -> chess.Move:
        """Convert a policy index to a chess move.

        Args:
            index: Index in the policy output.

        Returns:
            Chess move.
        """
        legal_moves = list(self.board.legal_moves)
        return legal_moves[index]
