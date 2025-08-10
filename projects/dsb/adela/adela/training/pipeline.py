"""Training pipeline for the chess AI."""

import os
from pathlib import Path
from typing import final, override
from collections.abc import Iterator

import chess
import chess.pgn
import numpy as np
import polars as pl
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm

from adela.core.board import BoardRepresentation
from adela.experts.base import ExpertBase
from adela.experts.specialized import (
    create_phase_experts,
    create_style_experts,
    create_adaptation_experts,
)
from adela.gating.system import MixtureOfExperts


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
    
    @override
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
        min_elo: int = 1000,
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

    def process_game(self, game: chess.pgn.Game) -> tuple[list[str], list[np.ndarray], list[float]]:
        """Public wrapper to process a single game into training triples."""
        return self._process_game(game)


@final
class ParquetProcessor:
    """Processor for Parquet data to extract training data.

    Uses Polars to load Parquet rows, ensures a parsed SAN move list is present,
    and converts each game into training positions/policies/values similar to
    the PGN processor.
    """

    def __init__(
        self,
        min_elo: int = 1000,
        max_positions_per_game: int = 30,
    ) -> None:
        self.min_elo = min_elo
        self.max_positions_per_game = max_positions_per_game

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

    def process_row(
        self,
        row: dict[str, object],
        result_col: str | None,
        parsed_moves_col: str,
    ) -> tuple[list[str], list[np.ndarray], list[float]]:
        """Public wrapper to process a single row into training triples."""
        return self._process_row(row, result_col, parsed_moves_col)

    def process_parquet(self, parquet_path: str) -> tuple[list[str], list[np.ndarray], list[float]]:
        """Process a Parquet file (or directory of files) into training data.

        Args:
            parquet_path: Path to a .parquet file or a directory containing parquet files.

        Returns:
            positions, policies, values
        """
        path = os.fspath(parquet_path)

        # Accumulators for all positions/policies/values
        positions_all: list[str] = []
        policies_all: list[np.ndarray] = []
        values_all: list[float] = []

        # Load data: support single file or directory. For directories, process file-by-file to
        # avoid a large collect() that can appear to hang on big datasets and to provide progress logs.
        if os.path.isdir(path):
            files = sorted(Path(path).glob("*.parquet"))
            if not files:
                return [], [], []

            print(f"[ParquetProcessor] Found {len(files)} parquet files in {path}")
            for idx, fp in enumerate(files):
                try:
                    df = pl.read_parquet(str(fp))
                except Exception as err:  # pragma: no cover - defensive logging
                    print(f"[ParquetProcessor] Skipping '{fp.name}' (read error): {err}")
                    continue

                # Elo filter per file
                df = self._filter_by_elo(df)
                if len(df) == 0:
                    if (idx + 1) % max(1, len(files) // 10) == 0 or idx == len(files) - 1:
                        print(f"[ParquetProcessor] {fp.name}: 0 rows after Elo filter; processed {idx+1}/{len(files)} files, positions so far: {len(positions_all)}")
                    continue

                # Determine required columns
                parsed_moves_col = self._pick_column(df, ["parsed_moves"])
                if not parsed_moves_col or parsed_moves_col not in df.columns:
                    print(f"[ParquetProcessor] {fp.name}: missing 'parsed_moves' column; skipping")
                    continue
                result_col = self._pick_column(df, ["result", "Result"]) if df is not None else None

                cols: list[str] = [parsed_moves_col] + ([result_col] if result_col else [])
                for row in df.select(cols).to_dicts():
                    pos, pol, val = self._process_row(row, result_col, parsed_moves_col)
                    if pos:
                        positions_all.extend(pos)
                        policies_all.extend(pol)
                        values_all.extend(val)

                # Periodic progress update
                if (idx + 1) % max(1, len(files) // 10) == 0 or idx == len(files) - 1:
                    print(f"[ParquetProcessor] Processed {idx+1}/{len(files)} files; positions so far: {len(positions_all)}")

        else:
            # Single parquet file path
            df = pl.read_parquet(path)

            # Elo filter
            df = self._filter_by_elo(df)
            if len(df) == 0:
                return [], [], []

            # Require parsed_moves to exist
            parsed_moves_col = self._pick_column(df, ["parsed_moves"])
            if not parsed_moves_col or parsed_moves_col not in df.columns:
                raise ValueError("Parquet data missing required 'parsed_moves' column. Data must be pre-processed.")

            result_col = self._pick_column(df, ["result", "Result"]) if df is not None else None

            cols: list[str] = [parsed_moves_col] + ([result_col] if result_col else [])
            for row in df.select(cols).to_dicts():
                pos, pol, val = self._process_row(row, result_col, parsed_moves_col)
                if pos:
                    positions_all.extend(pos)
                    policies_all.extend(pol)
                    values_all.extend(val)

        return positions_all, policies_all, values_all


@final
class StreamingParquetDataset(IterableDataset[tuple[np.ndarray, np.ndarray, np.ndarray, float]]):
    """Stream training samples directly from Parquet files in small chunks.

    This avoids loading all data into memory by scanning each Parquet file and
    yielding per-position training samples on the fly.
    """

    def __init__(
        self,
        data_path: str | Path,
        min_elo: int = 1000,
        max_positions_per_game: int = 30,
        chunk_rows: int = 10_000,
        shuffle_files: bool = False,
    ) -> None:
        self.data_path: str = os.fspath(data_path)
        self.min_elo: int = min_elo
        self.max_positions_per_game: int = max_positions_per_game
        self.chunk_rows: int = max(1, int(chunk_rows))
        self.shuffle_files: bool = shuffle_files

        self._processor = ParquetProcessor(min_elo=min_elo, max_positions_per_game=max_positions_per_game)

        if os.path.isdir(self.data_path):
            self._files = sorted(Path(self.data_path).glob("*.parquet"))
        else:
            self._files = [Path(self.data_path)]

    def _pick_column(self, df: pl.DataFrame, candidates: list[str]) -> str | None:
        # Delegate to processor's public behavior through the same method
        return ParquetProcessor()._pick_column(df, candidates)

    def _filter_by_elo(self, df: pl.DataFrame) -> pl.DataFrame:
        # Use a temporary processor with the configured min_elo
        tmp = ParquetProcessor(min_elo=self.min_elo, max_positions_per_game=self.max_positions_per_game)
        return tmp._filter_by_elo(df)

    def _rows_iter(self, file_path: Path) -> Iterator[dict[str, object]]:
        # Determine columns early to avoid reading unnecessary data
        # We still need to collect a small slice to infer available columns robustly
        try:
            lazy = pl.scan_parquet(str(file_path))
            # Count total rows lazily
            total_rows = lazy.select(pl.count()).collect().item()
        except Exception:
            return iter(())

        start = 0
        while start < total_rows:
            try:
                df = (
                    pl.scan_parquet(str(file_path))
                    .slice(start, self.chunk_rows)
                    .collect()
                )
            except Exception:
                break

            if len(df) == 0:
                break

            df = self._filter_by_elo(df)
            if len(df) == 0:
                start += self.chunk_rows
                continue

            parsed_moves_col = self._pick_column(df, ["parsed_moves"]) or "parsed_moves"
            result_col = self._pick_column(df, ["result", "Result"])  # may be None

            cols: list[str] = [parsed_moves_col] + ([result_col] if result_col else [])
            # Some files might be missing a requested column; guard selection
            cols = [c for c in cols if c in df.columns]
            if not cols or parsed_moves_col not in cols:
                start += self.chunk_rows
                continue

            for row in df.select(cols).to_dicts():
                yield row

            start += self.chunk_rows

    @override
    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        files = list(self._files)
        if self.shuffle_files:
            try:
                import random as _random  # local import to avoid global dependency
                _random.shuffle(files)
            except Exception:
                pass

        for fp in files:
            for row in self._rows_iter(fp):
                # Determine column names directly from row keys
                result_col = "result" if "result" in row else ("Result" if "Result" in row else None)
                parsed_moves_col = "parsed_moves"

                positions, policies, values = self._processor.process_row(
                    row=row,
                    result_col=result_col,
                    parsed_moves_col=parsed_moves_col,
                )
                if not positions:
                    continue

                for fen, policy, value in zip(positions, policies, values):
                    board = BoardRepresentation(fen)
                    board_tensor = board.get_board_tensor()
                    additional_features = board.get_additional_features()
                    yield board_tensor, additional_features, policy, value


@final
class StreamingPGNDataset(IterableDataset[tuple[np.ndarray, np.ndarray, np.ndarray, float]]):
    """Stream training samples directly from a PGN file.

    Opens the PGN file each iteration and yields per-position samples without
    caching everything in memory.
    """

    def __init__(
        self,
        pgn_path: str | Path,
        min_elo: int = 1000,
        max_positions_per_game: int = 30,
    ) -> None:
        self.pgn_path: str = os.fspath(pgn_path)
        self._processor = PGNProcessor(min_elo=min_elo, max_positions_per_game=max_positions_per_game)

    @override
    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        with open(self.pgn_path, "r") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                # Filter by Elo inside processor logic
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
                if white_elo < self._processor.min_elo or black_elo < self._processor.min_elo:
                    continue

                positions, policies, values = self._processor.process_game(game)
                for fen, policy, value in zip(positions, policies, values):
                    board = BoardRepresentation(fen)
                    board_tensor = board.get_board_tensor()
                    additional_features = board.get_additional_features()
                    yield board_tensor, additional_features, policy, value


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
        _ = self.model.to(self.device)
        
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
        _ = self.model.train()

        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        batch_count = 0

        # Progress bar over batches; len(dataloader) may be unavailable for IterableDataset
        try:
            total_batches = len(dataloader)
        except Exception:
            total_batches = None

        pbar = tqdm(dataloader, total=total_batches, desc="train", unit="batch")

        for batch in pbar:
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
            _ = loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            batch_count += 1

            # Update progress display with current and running average losses
            avg_loss = total_loss / max(1, batch_count)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{avg_loss:.4f}",
            })
        
        # Calculate average metrics
        num_batches = batch_count if batch_count > 0 else 1
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
        _ = self.model.eval()

        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        batch_count = 0

        # Progress bar over validation; len(dataloader) may be unavailable for IterableDataset
        try:
            total_batches = len(dataloader)
        except Exception:
            total_batches = None

        with torch.no_grad():
            pbar = tqdm(dataloader, total=total_batches, desc="val", unit="batch")
            for batch in pbar:
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
                batch_count += 1

                avg_loss = total_loss / max(1, batch_count)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{avg_loss:.4f}",
                })
        
        # Calculate average metrics
        num_batches = batch_count if batch_count > 0 else 1
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
        _ = self.model.load_state_dict(torch.load(path, map_location=self.device))


def collate_training_batch(
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function to assemble a training batch with correct dtypes.

    - Stacks board tensors and additional features as float32
    - Converts policy one-hot/probability vectors to class indices (long)
    - Assembles value targets as float32
    """
    boards_np, feats_np, policies_arr, values_list = zip(*batch)

    boards = torch.from_numpy(np.stack(boards_np)).to(torch.float32)
    feats = torch.from_numpy(np.stack(feats_np)).to(torch.float32)

    # Policies can be numpy arrays or tensors. Normalize to float tensor then to class indices.
    if isinstance(policies_arr[0], np.ndarray):
        policy_tensor = torch.from_numpy(np.stack(policies_arr)).to(torch.float32)
    elif isinstance(policies_arr[0], torch.Tensor):
        policy_tensor = torch.stack([p.to(torch.float32) for p in policies_arr])
    else:
        # Fallback: convert via numpy
        policy_tensor = torch.from_numpy(np.stack([np.asarray(p) for p in policies_arr])).to(torch.float32)

    # Convert to class indices for CrossEntropyLoss
    if policy_tensor.ndim == 2:
        policy_target = policy_tensor.argmax(dim=1).to(torch.long)
    else:
        policy_target = policy_tensor.to(torch.long)

    value_target = torch.tensor(values_list, dtype=torch.float32)

    return boards, feats, policy_target, value_target


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
    _ = model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    return model
