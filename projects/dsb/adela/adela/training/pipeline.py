"""Training pipeline for the chess AI."""

import os
import time
from typing import Dict, List, Tuple, Any, Optional, Iterator

import chess
import chess.pgn
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from adela.core.board import BoardRepresentation
from adela.experts.base import ExpertBase
from adela.experts.specialized import (
    create_phase_experts,
    create_style_experts,
    create_adaptation_experts,
)
from adela.gating.system import MixtureOfExperts
from adela.mcts.search import MCTS


class ChessDataset(Dataset):
    """Dataset for training chess models."""

    def __init__(
        self, 
        positions: List[str],
        policies: List[np.ndarray],
        values: List[float],
        augment: bool = True
    ) -> None:
        """Initialize the dataset.

        Args:
            positions: List of FEN strings.
            policies: List of policy vectors.
            values: List of position values.
            augment: Whether to augment the data with board flips and rotations.
        """
        self.positions = positions
        self.policies = policies
        self.values = values
        self.augment = augment
        
    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            Dataset length.
        """
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get an item from the dataset.

        Args:
            idx: Index of the item.

        Returns:
            Tuple of (board_tensor, additional_features, policy, value).
        """
        # Get the position
        fen = self.positions[idx]
        board = BoardRepresentation(fen)
        
        # Get the board tensor and additional features
        board_tensor = board.get_board_tensor()
        additional_features = board.get_additional_features()
        
        # Get the policy and value
        policy = self.policies[idx]
        value = self.values[idx]
        
        return board_tensor, additional_features, policy, value


class PGNProcessor:
    """Processor for PGN files to extract training data."""

    def __init__(
        self, 
        min_elo: int = 2000,
        max_positions_per_game: int = 30
    ) -> None:
        """Initialize the PGN processor.

        Args:
            min_elo: Minimum Elo rating for games to include.
            max_positions_per_game: Maximum number of positions to extract per game.
        """
        self.min_elo = min_elo
        self.max_positions_per_game = max_positions_per_game
        
    def process_pgn_file(
        self, 
        pgn_path: str
    ) -> Tuple[List[str], List[np.ndarray], List[float]]:
        """Process a PGN file to extract training data.

        Args:
            pgn_path: Path to the PGN file.

        Returns:
            Tuple of (positions, policies, values).
        """
        positions = []
        policies = []
        values = []
        
        # Open the PGN file
        with open(pgn_path, "r") as f:
            while True:
                # Read the next game
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                # Check Elo ratings
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
                if white_elo < self.min_elo or black_elo < self.min_elo:
                    continue
                
                # Process the game
                game_positions, game_policies, game_values = self._process_game(game)
                
                positions.extend(game_positions)
                policies.extend(game_policies)
                values.extend(game_values)
        
        return positions, policies, values
    
    def _process_game(
        self, 
        game: chess.pgn.Game
    ) -> Tuple[List[str], List[np.ndarray], List[float]]:
        """Process a single game to extract training data.

        Args:
            game: Chess game.

        Returns:
            Tuple of (positions, policies, values).
        """
        positions = []
        policies = []
        values = []
        
        # Get the result
        result = game.headers.get("Result", "*")
        if result == "1-0":
            game_value = 1.0
        elif result == "0-1":
            game_value = -1.0
        else:
            game_value = 0.0
        
        # Initialize the board
        board = game.board()
        board_rep = BoardRepresentation(board.fen())
        
        # Process moves
        moves = list(game.mainline_moves())
        num_positions = min(len(moves), self.max_positions_per_game)
        
        for i in range(num_positions):
            # Get the current position
            fen = board.fen()
            
            # Create policy vector
            policy = np.zeros(1968, dtype=np.float32)  # Maximum possible moves
            
            # Set the policy for the actual move played
            move = moves[i]
            move_idx = board_rep.get_move_index(move)
            policy[move_idx] = 1.0
            
            # Add the position, policy, and value
            positions.append(fen)
            policies.append(policy)
            
            # Value from the perspective of the current player
            value = game_value if board.turn == chess.WHITE else -game_value
            values.append(value)
            
            # Make the move
            board.push(move)
            board_rep = BoardRepresentation(board.fen())
        
        return positions, policies, values


class SelfPlayGenerator:
    """Generator for self-play games."""

    def __init__(
        self, 
        model: MixtureOfExperts,
        num_games: int = 1000,
        mcts_simulations: int = 800,
        temperature: float = 1.0
    ) -> None:
        """Initialize the self-play generator.

        Args:
            model: Neural network model.
            num_games: Number of games to generate.
            mcts_simulations: Number of MCTS simulations per move.
            temperature: Temperature for move selection.
        """
        self.model = model
        self.num_games = num_games
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        
        # Create MCTS
        self.mcts = MCTS(
            model=model,
            num_simulations=mcts_simulations,
            temperature=temperature
        )
        
    def generate_games(self) -> Tuple[List[str], List[np.ndarray], List[float]]:
        """Generate self-play games.

        Returns:
            Tuple of (positions, policies, values).
        """
        positions = []
        policies = []
        values = []
        
        for _ in range(self.num_games):
            # Generate a game
            game_positions, game_policies, game_values = self._generate_game()
            
            positions.extend(game_positions)
            policies.extend(game_policies)
            values.extend(game_values)
        
        return positions, policies, values
    
    def _generate_game(self) -> Tuple[List[str], List[np.ndarray], List[float]]:
        """Generate a single self-play game.

        Returns:
            Tuple of (positions, policies, values).
        """
        positions = []
        policies = []
        values = []
        
        # Initialize the board
        board = BoardRepresentation()
        
        # Play the game
        while not board.is_game_over():
            # Get the current position
            fen = board.get_fen()
            
            # Run MCTS
            visit_counts = self.mcts.search(board)
            
            # Create policy vector from visit counts
            policy = np.zeros(1968, dtype=np.float32)  # Maximum possible moves
            
            # Set the policy based on visit counts
            legal_moves = board.get_legal_moves()
            visit_sum = sum(visit_counts.values())
            
            for move in legal_moves:
                if move in visit_counts:
                    move_idx = board.get_move_index(move)
                    policy[move_idx] = visit_counts[move] / visit_sum
            
            # Add the position and policy
            positions.append(fen)
            policies.append(policy)
            
            # Select move based on visit counts and temperature
            move = self.mcts.get_best_move(board, self.temperature)
            
            # Make the move
            board.make_move(move)
        
        # Get the game result
        result = board.get_result()
        if result == "1-0":
            game_value = 1.0
        elif result == "0-1":
            game_value = -1.0
        else:
            game_value = 0.0
        
        # Set the values for all positions
        for i, fen in enumerate(positions):
            # Value from the perspective of the current player
            board_temp = chess.Board(fen)
            value = game_value if board_temp.turn == chess.WHITE else -game_value
            values.append(value)
        
        return positions, policies, values


class Trainer:
    """Trainer for the chess AI."""

    def __init__(
        self, 
        model: MixtureOfExperts,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Neural network model.
            lr: Learning rate.
            weight_decay: Weight decay.
            batch_size: Batch size.
            device: Device to train on.
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        
        # Move model to device
        self.model.to(device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
    def train_epoch(
        self, 
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: DataLoader for training data.

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        for batch in dataloader:
            # Get batch data
            board_tensor, additional_features, policy_target, value_target = batch
            
            # Move to device
            board_tensor = board_tensor.to(self.device)
            additional_features = additional_features.to(self.device)
            policy_target = policy_target.to(self.device)
            value_target = value_target.to(self.device)
            
            # Forward pass
            policy_logits, value, _ = self.model(board_tensor, additional_features)
            
            # Calculate loss
            policy_loss = F.cross_entropy(policy_logits, policy_target)
            value_loss = F.mse_loss(value.squeeze(-1), value_target)
            
            # Combined loss
            loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
        
        # Calculate average metrics
        num_batches = len(dataloader)
        metrics = {
            "loss": total_loss / num_batches,
            "policy_loss": policy_loss_sum / num_batches,
            "value_loss": value_loss_sum / num_batches
        }
        
        return metrics
    
    def validate(
        self, 
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Validate the model.

        Args:
            dataloader: DataLoader for validation data.

        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                # Get batch data
                board_tensor, additional_features, policy_target, value_target = batch
                
                # Move to device
                board_tensor = board_tensor.to(self.device)
                additional_features = additional_features.to(self.device)
                policy_target = policy_target.to(self.device)
                value_target = value_target.to(self.device)
                
                # Forward pass
                policy_logits, value, _ = self.model(board_tensor, additional_features)
                
                # Calculate loss
                policy_loss = F.cross_entropy(policy_logits, policy_target)
                value_loss = F.mse_loss(value.squeeze(-1), value_target)
                
                # Combined loss
                loss = policy_loss + value_loss
                
                # Update metrics
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
        
        # Calculate average metrics
        num_batches = len(dataloader)
        metrics = {
            "val_loss": total_loss / num_batches,
            "val_policy_loss": policy_loss_sum / num_batches,
            "val_value_loss": value_loss_sum / num_batches
        }
        
        return metrics
    
    def save_model(self, path: str) -> None:
        """Save the model.

        Args:
            path: Path to save the model.
        """
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str) -> None:
        """Load the model.

        Args:
            path: Path to load the model from.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def create_mixture_of_experts(
    num_filters: int = 256,
    num_blocks: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> MixtureOfExperts:
    """Create a mixture of experts model.

    Args:
        num_filters: Number of filters in convolutional layers.
        num_blocks: Number of residual blocks.
        device: Device to create the model on.

    Returns:
        Mixture of experts model.
    """
    # Create experts
    phase_experts = create_phase_experts(num_filters, num_blocks)
    style_experts = create_style_experts(num_filters, num_blocks)
    adaptation_experts = create_adaptation_experts(num_filters, num_blocks)
    
    # Combine experts
    experts = {}
    experts.update(phase_experts)
    experts.update(style_experts)
    experts.update(adaptation_experts)
    
    # Create mixture of experts
    model = MixtureOfExperts(experts)
    
    # Move to device
    model.to(device)
    
    return model
