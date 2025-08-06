"""Tests for the board representation."""

import chess
import numpy as np
import pytest

from adela.core.board import BoardRepresentation


def test_board_initialization():
    """Test board initialization."""
    # Default initialization
    board = BoardRepresentation()
    assert board.board.fen().startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    
    # Custom FEN
    custom_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    board = BoardRepresentation(custom_fen)
    assert board.board.fen() == custom_fen


def test_legal_moves():
    """Test getting legal moves."""
    board = BoardRepresentation()
    legal_moves = board.get_legal_moves()
    assert len(legal_moves) == 20  # 20 legal moves in the starting position


def test_make_move():
    """Test making a move."""
    board = BoardRepresentation()
    # e2e4
    move = chess.Move.from_uci("e2e4")
    board.make_move(move)
    assert board.board.fen().startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")


def test_undo_move():
    """Test undoing a move."""
    board = BoardRepresentation()
    initial_fen = board.board.fen()
    
    # Make a move
    move = chess.Move.from_uci("e2e4")
    board.make_move(move)
    
    # Undo the move
    board.undo_move()
    assert board.board.fen() == initial_fen


def test_game_over():
    """Test game over detection."""
    # Starting position is not game over
    board = BoardRepresentation()
    assert not board.is_game_over()
    
    # Fool's mate is game over
    board = BoardRepresentation("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_game_over()
    assert board.get_result() == "0-1"


def test_board_tensor():
    """Test board tensor representation."""
    board = BoardRepresentation()
    tensor = board.get_board_tensor()
    
    # Check shape
    assert tensor.shape == (12, 8, 8)
    
    # Check white pawns
    assert np.sum(tensor[0]) == 8  # 8 white pawns
    
    # Check black pawns
    assert np.sum(tensor[6]) == 8  # 8 black pawns


def test_additional_features():
    """Test additional features."""
    board = BoardRepresentation()
    features = board.get_additional_features()
    
    # Check shape
    assert len(features) == 8
    
    # Check side to move
    assert features[0] == 1.0  # White to move
    
    # Check castling rights
    assert features[1] == 1.0  # White kingside
    assert features[2] == 1.0  # White queenside
    assert features[3] == 1.0  # Black kingside
    assert features[4] == 1.0  # Black queenside


def test_get_phase():
    """Test game phase calculation."""
    # Starting position
    board = BoardRepresentation()
    phase = board.get_phase()
    assert 0.0 <= phase <= 0.1  # Close to opening
    
    # Middle game (some pieces captured)
    board = BoardRepresentation("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 5")
    phase = board.get_phase()
    assert 0.1 <= phase <= 0.5  # Middle game
    
    # Endgame (few pieces left)
    board = BoardRepresentation("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    phase = board.get_phase()
    assert 0.8 <= phase <= 1.0  # Close to endgame
