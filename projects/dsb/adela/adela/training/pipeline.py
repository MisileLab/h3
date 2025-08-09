"""Training pipeline for the chess AI."""

import os
from pathlib import Path
from typing import final

import chess
import chess.pgn
import numpy as np
import polars as pl
import torch
import torch.optim as optim
import torch.nn.functional as F
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
from load_and_parse_parquets import add_parsed_moves


@final
class ChessDataset(Dataset[tuple[np.ndarray, np.ndarray, np.ndarray, float]]):
    """Dataset for training chess models."""

    def __init__(
        self,
        positions: list[str],
        policies: list[np.ndarray],
        values: list[float],
        augment: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            positions: List of FEN strings.
            policies: List of policy vectors.
            values: List of position values.
            augment: Whether to augment the data with board flips and rotations.
        """
        self.positions: list[str] = positions
        self.policies: list[np.ndarray] = policies
        self.values: list[float] = values
        self.augment: bool = augment
        
    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            Dataset length.
        """
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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


@final
class PGNProcessor:
    """Processor for PGN files to extract training data."""

    def __init__(
        self,
        min_elo: int = 2000,
        max_positions_per_game: int = 30,
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
        pgn_path: str,
    ) -> tuple[list[str], list[np.ndarray], list[float]]:
        """Process a PGN file to extract training data.

        Args:
            pgn_path: Path to the PGN file.

        Returns:
            Tuple of (positions, policies, values).
        """
        positions: list[str] = []
        policies: list[np.ndarray] = []
        values: list[float] = []
        
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
        game: chess.pgn.Game,
    ) -> tuple[list[str], list[np.ndarray], list[float]]:
        """Process a single game to extract training data.

        Args:
            game: Chess game.

        Returns:
            Tuple of (positions, policies, values).
        """
        positions: list[str] = []
        policies: list[np.ndarray] = []
        values: list[float] = []
        
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


@final
class ParquetProcessor:
    """Processor for Parquet data to extract training data.

    Uses Polars to load Parquet rows, ensures a parsed SAN move list is present,
    and converts each game into training positions/policies/values similar to
    the PGN processor.
    """

    def __init__(
        self,
        min_elo: int = 2000,
        max_positions_per_game: int = 30,
        parse_chunk_size: int = 1000,
    ) -> None:
        self.min_elo = min_elo
        self.max_positions_per_game = max_positions_per_game
        self.parse_chunk_size = parse_chunk_size

    def _pick_column(self, df: pl.DataFrame, candidates: list[str]) -> str | None:
        for name in candidates:
            if name in df.columns:
                return name
        # try case-insensitive
        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    def _filter_by_elo(self, df: pl.DataFrame) -> pl.DataFrame:
        white_col = self._pick_column(df, ["WhiteElo", "white_elo"]) or "WhiteElo"
        black_col = self._pick_column(df, ["BlackElo", "black_elo"]) or "BlackElo"
        if white_col in df.columns and black_col in df.columns:
            return df.filter((pl.col(white_col) >= self.min_elo) | (pl.col(black_col) >= self.min_elo))
        return df

    def _game_value_from_result(self, result: str) -> float:
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return -1.0
        return 0.0

    def _process_row(
        self,
        row: dict[str, object],
        result_col: str | None,
        parsed_moves_col: str,
    ) -> tuple[list[str], list[np.ndarray], list[float]]:
        positions: list[str] = []
        policies: list[np.ndarray] = []
        values: list[float] = []

        result_str = (row.get(result_col) if result_col else None) or row.get("Result") or row.get("result") or "*"
        game_value = self._game_value_from_result(str(result_str))

        board = chess.Board()
        board_rep = BoardRepresentation(board.fen())

        raw_val = row.get(parsed_moves_col)
        moves_san: list[str]
        if isinstance(raw_val, list):
            moves_san = [str(m) for m in raw_val]
        elif isinstance(raw_val, str):
            # Some pipelines might store parsed_moves as a space-joined string; handle both
            moves_san = [m for m in raw_val.split() if m]
        else:
            moves_san = []

        limit = min(len(moves_san), self.max_positions_per_game)
        for i in range(limit):
            fen = board.fen()

            policy = np.zeros(1968, dtype=np.float32)
            san = moves_san[i]
            try:
                move = board.parse_san(san)
            except ValueError:
                # Skip unparsable SAN
                break

            try:
                move_idx = board_rep.get_move_index(move)
            except ValueError:
                # If mapping fails, skip
                break

            policy[move_idx] = 1.0
            positions.append(fen)
            policies.append(policy)
            values.append(game_value if board.turn == chess.WHITE else -game_value)

            board.push(move)
            board_rep = BoardRepresentation(board.fen())

        return positions, policies, values

    def process_parquet(self, parquet_path: str) -> tuple[list[str], list[np.ndarray], list[float]]:
        """Process a Parquet file (or directory of files) into training data.

        Args:
            parquet_path: Path to a .parquet file or a directory containing parquet files.

        Returns:
            positions, policies, values
        """
        path = os.fspath(parquet_path)

        # Load data: support single file or directory
        if os.path.isdir(path):
            # Lazy scan and collect for all parquet files
            lazy = pl.concat([pl.scan_parquet(str(p)) for p in sorted(Path(path).glob("*.parquet"))])
            df = lazy.collect()
        else:
            df = pl.read_parquet(path)

        # Elo filter
        df = self._filter_by_elo(df)
        if len(df) == 0:
            return [], [], []

        # Ensure parsed_moves exist
        parsed_moves_col = self._pick_column(df, ["parsed_moves"]) or "parsed_moves"
        if parsed_moves_col not in df.columns:
            df = add_parsed_moves(df, chunk_size=self.parse_chunk_size)
            parsed_moves_col = "parsed_moves"

        result_col = self._pick_column(df, ["result", "Result"]) if df is not None else None

        positions_all: list[str] = []
        policies_all: list[np.ndarray] = []
        values_all: list[float] = []

        cols: list[str] = [parsed_moves_col] + ([result_col] if result_col else [])
        for row in df.select(cols).to_dicts():
            pos, pol, val = self._process_row(row, result_col, parsed_moves_col)
            if pos:
                positions_all.extend(pos)
                policies_all.extend(pol)
                values_all.extend(val)

        return positions_all, policies_all, values_all

@final
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
        
    def generate_games(self) -> tuple[list[str], list[np.ndarray], list[float]]:
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
    
    def _generate_game(self) -> tuple[list[str], list[np.ndarray], list[float]]:
        """Generate a single self-play game.

        Returns:
            Tuple of (positions, policies, values).
        """
        positions: list[str] = []
        policies: list[np.ndarray] = []
        values: list[float] = []
        
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
        for fen in positions:
            # Value from the perspective of the current player
            board_temp = chess.Board(fen)
            value = game_value if board_temp.turn == chess.WHITE else -game_value
            values.append(value)
        
        return positions, policies, values


@final
class Trainer:
    """Trainer for the chess AI."""

    def __init__(
        self,
        model: MixtureOfExperts,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        device: str | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Neural network model.
            lr: Learning rate.
            weight_decay: Weight decay.
            batch_size: Batch size.
            device: Device to train on.
        """
        self.model: MixtureOfExperts = model
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size: int = batch_size
        
        # Move model to device
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer: optim.Optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
    def train_epoch(
        self,
        dataloader: DataLoader[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    ) -> dict[str, float]:
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
        metrics: dict[str, float] = {
            "loss": total_loss / num_batches,
            "policy_loss": policy_loss_sum / num_batches,
            "value_loss": value_loss_sum / num_batches
        }
        
        return metrics
    
    def validate(
        self,
        dataloader: DataLoader[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    ) -> dict[str, float]:
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
        metrics: dict[str, float] = {
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
    device: str | None = None,
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
    experts: dict[str, ExpertBase] = {}
    experts.update(phase_experts)
    experts.update(style_experts)
    experts.update(adaptation_experts)
    
    # Create mixture of experts
    model = MixtureOfExperts(experts)
    
    # Move to device
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    return model
