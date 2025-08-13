"""Self-play training script with batched Parquet output.

This script generates self-play games using the current Mixture-of-Experts
model with MCTS, writes per-position training samples to multiple Parquet
files, then trains the model by streaming those samples.

Parquet schema for self-play samples:
- fen: str
- policy: list[float] of length 1968 (sparse: non-zeros only at legal indices)
- value: float in [-1.0, 0.0, 1.0]

Usage (example):
  # One-shot generate-then-train
  python -m scripts.train_self_play \
      --output-data data/selfplay \
      --output-models models/selfplay \
      --num-games 200 \
      --simulations 200 \
      --chunk-size 5000 \
      --epochs 5

  # Continuous loop until Ctrl+C
  python -m scripts.train_self_play \
      --output-data data/selfplay \
      --output-models models/selfplay \
      --simulations 200 \
      --games-per-iter 50 \
      --epochs-per-iter 1 \
      --until-interrupt
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator, Mapping
from typing import Optional

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

from adela.core.board import BoardRepresentation
from adela.mcts.search import MCTS
from adela.training.pipeline import (
    Trainer,
    collate_training_batch,
    create_mixture_of_experts,
)
from adela.gating.system import MixtureOfExperts


# ----------------------------
# Data generation (self-play)
# ----------------------------

@dataclass
class SelfPlayConfig:
    num_games: int = 100
    max_moves_per_game: int = 200
    simulations_per_move: int = 200
    temperature: float = 1.0
    chunk_size: int = 10_000
    mcts_batch_size: int = 16,
    output_dir: Path = Path("data/selfplay")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _policy_from_visit_counts(
    board: BoardRepresentation,
    visit_counts: Mapping[object, float]
) -> np.ndarray:
    """Convert MCTS visit counts to a 1968-dim policy distribution.

    The policy is zero everywhere except at legal move indices for the current
    position, where it stores the normalized visit counts.
    """
    legal_moves = board.get_legal_moves()
    counts = np.array([float(visit_counts.get(m, 0)) for m in legal_moves], dtype=np.float32)
    total = float(counts.sum())
    if total <= 0.0:
        # Uniform fallback if search returned no visits (shouldn't happen with >0 simulations)
        probs = np.full(len(legal_moves), 1.0 / max(1, len(legal_moves)), dtype=np.float32)
    else:
        probs = counts / total

    policy = np.zeros(1968, dtype=np.float32)
    for i, move in enumerate(legal_moves):
        try:
            idx = board.get_move_index(move)
        except Exception:
            # If index mapping fails, skip this move
            continue
        if 0 <= idx < policy.shape[0]:
            policy[idx] = probs[i]
    return policy


def _sample_move_from_counts(
    legal_moves: list[object], counts: np.ndarray, temperature: float
) -> object:
    """Sample a move from visit counts with temperature."""
    if len(legal_moves) == 0:
        raise RuntimeError("No legal moves available for sampling.")

    if temperature <= 1e-6:
        return legal_moves[int(np.argmax(counts))]

    probs = counts.astype(np.float64)
    if probs.sum() <= 0:
        probs = np.ones_like(probs, dtype=np.float64)
    probs = np.power(probs, 1.0 / float(temperature))
    probs = probs / probs.sum()
    idx = int(np.random.choice(np.arange(len(legal_moves)), p=probs))
    return legal_moves[idx]


def _compute_last_batch_index(output_dir: Path) -> int:
    """Return the highest existing selfplay batch index in the output directory.

    If none exist, return 0.
    """
    if not output_dir.exists():
        return 0
    last = 0
    for fp in output_dir.glob("selfplay_batch_*.parquet"):
        name = fp.name
        try:
            # Expect selfplay_batch_000123.parquet
            num_str = name.split("selfplay_batch_")[1].split(".parquet")[0]
            idx = int(num_str)
            last = max(last, idx)
        except Exception:
            continue
    return last


def _run_single_self_play_game(model: MixtureOfExperts, cfg: SelfPlayConfig) -> list[dict[str, object]]:
    """Runs a single self-play game and returns the generated rows."""
    mcts = MCTS(
        model=model, num_simulations=cfg.simulations_per_move, temperature=cfg.temperature, device=cfg.device, batch_size=cfg.mcts_batch_size
    )

    board = BoardRepresentation()
    fens: list[str] = []
    policies: list[np.ndarray] = []
    side_to_move_is_white: list[bool] = []

    move_count = 0
    while not board.is_game_over() and move_count < cfg.max_moves_per_game:
        visit_counts = mcts.search(board)
        policy = _policy_from_visit_counts(board, visit_counts)

        fens.append(board.get_fen())
        policies.append(policy)
        side_to_move_is_white.append(bool(board.board.turn))

        legal_moves = board.get_legal_moves()
        counts = np.array([float(visit_counts.get(m, 0)) for m in legal_moves], dtype=np.float32)
        move = _sample_move_from_counts(legal_moves, counts, cfg.temperature)
        board.make_move(move)
        move_count += 1

    result = board.get_result()
    if result == "1-0":
        game_value = 1.0
    elif result == "0-1":
        game_value = -1.0
    else:
        game_value = 0.0

    values = [game_value if is_white else -game_value for is_white in side_to_move_is_white]

    game_rows: list[dict[str, object]] = []
    for fen, pol, val in zip(fens, policies, values):
        game_rows.append({"fen": fen, "policy": pol.tolist(), "value": float(val)})

    return game_rows

def generate_self_play_data(
    model: MixtureOfExperts, cfg: SelfPlayConfig, *, last_batch_index: Optional[int] = None
) -> list[Path]:
    """Generate self-play data and write to multiple Parquet files.

    Returns list of written parquet file paths.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    files: list[Path] = []
    if last_batch_index is None:
        last_batch_index = _compute_last_batch_index(cfg.output_dir)
    batch_idx = int(last_batch_index)

    game_args = [(model, cfg) for _ in range(cfg.num_games)]

    num_threads = os.cpu_count()
    print(f"Generating {cfg.num_games} games using {num_threads} threads...")

    all_game_rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(lambda p: _run_single_self_play_game(*p), game_args),
                              total=cfg.num_games, desc="Generating self-play games"))
    all_game_rows = [row for result in results for row in result]

    rows_to_write = all_game_rows
    current_row_idx = 0
    while current_row_idx < len(rows_to_write):
        chunk = rows_to_write[current_row_idx : current_row_idx + cfg.chunk_size]
        if chunk:
            batch_idx += 1
            out_path = cfg.output_dir / f"selfplay_batch_{batch_idx:06d}.parquet"
            df = pl.DataFrame({
                "fen": [r["fen"] for r in chunk],
                "policy": [r["policy"] for r in chunk],
                "value": [r["value"] for r in chunk],
            })
            try:
                df.write_parquet(str(out_path), compression="zstd")
            except Exception:
                df.write_parquet(str(out_path))
            files.append(out_path)
        current_row_idx += cfg.chunk_size

    return files


# ------------------------------------
# Streaming dataset over self-play data
# ------------------------------------

class StreamingSelfPlayDataset(IterableDataset[tuple[np.ndarray, np.ndarray, np.ndarray, float]]):
    """Stream samples from self-play parquet files.

    Expects parquet files with columns: fen (str), policy (list[float]), value (float)
    """

    def __init__(self, data_dir_or_file: str | Path) -> None:
        super().__init__()
        path = Path(data_dir_or_file)
        self._files: list[Path] = []
        if path.is_dir():
            self._files = sorted(path.glob("*.parquet"))
        else:
            self._files = [path]

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        for fp in self._files:
            try:
                lazy = pl.scan_parquet(str(fp))
                total_rows = int(lazy.select(pl.len()).collect().item())
            except Exception:
                continue

            start = 0
            chunk_rows = 10_000
            while start < total_rows:
                try:
                    df = pl.scan_parquet(str(fp)).slice(start, chunk_rows).collect()
                except Exception:
                    break
                if len(df) == 0:
                    break

                # Ensure expected columns exist
                if not {"fen", "policy", "value"}.issubset(set(df.columns)):
                    break

                for row in df.iter_rows(named=True):
                    fen = str(row["fen"])  # type: ignore[arg-type]
                    policy_list = row["policy"]  # list[float]
                    value = float(row["value"])  # type: ignore[arg-type]

                    board = BoardRepresentation(fen)
                    board_tensor = board.get_board_tensor()
                    additional_features = board.get_additional_features()
                    policy = np.asarray(policy_list, dtype=np.float32)
                    yield board_tensor, additional_features, policy, value

                start += chunk_rows


# --------------
# Train function
# --------------

def train_from_selfplay(
    data_path: Path,
    output_dir: Path,
    num_epochs: int = 5,
    batch_size: int = 256,
    device: Optional[str] = None,
    model: Optional[MixtureOfExperts] = None,
) -> MixtureOfExperts:
    os.makedirs(output_dir, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model or create_mixture_of_experts(device=device)
    trainer = Trainer(model, batch_size=batch_size, device=device)

    # Split files for val if directory
    files = []
    if data_path.is_dir():
        files = sorted(data_path.glob("*.parquet"))
    else:
        files = [data_path]

    if len(files) >= 5:
        val_count = max(1, int(0.1 * len(files)))
        train_files = files[:-val_count]
        val_files = files[-val_count:]
    else:
        train_files = files
        val_files = []

    train_dataset = StreamingSelfPlayDataset(train_files[0].parent if data_path.is_dir() else data_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # NOTE: Set to 0 to avoid multiprocessing issues
        collate_fn=collate_training_batch,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = None
    if val_files:
        val_dataset = StreamingSelfPlayDataset(val_files[0].parent)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # NOTE: Set to 0 to avoid multiprocessing issues
            collate_fn=collate_training_batch,
            pin_memory=True if device == "cuda" else False,
        )

    best_metric = math.inf
    epochs_no_improve = 0
    patience = 3
    min_delta = 0.0
    best_ckpt_path = output_dir / "model_best.pt"

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_metrics = trainer.train_epoch(train_loader)
        print(f"Training loss: {train_metrics['loss']:.4f}")

        current_metric: float
        if val_loader is not None:
            val_metrics = trainer.validate(val_loader)
            current_metric = float(val_metrics["val_loss"])
            print(f"Validation loss: {current_metric:.4f}")
        else:
            current_metric = float(train_metrics["loss"])

        if best_metric - current_metric > min_delta:
            best_metric = current_metric
            epochs_no_improve = 0
            trainer.save_model(str(best_ckpt_path))
            print(f"Improved metric -> saved best: {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    final_model_path = output_dir / "model_final.pt"
    trainer.save_model(str(final_model_path))
    print(f"Saved final model: {final_model_path}")

    return model


# -------
# CLI
# -------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-play data generation and training")
    p.add_argument("--output-data", type=Path, default=Path("data/selfplay"), help="Directory to write parquet files")
    p.add_argument("--output-models", type=Path, default=Path("models/selfplay"), help="Directory to write model checkpoints")
    p.add_argument("--num-games", type=int, default=100, help="Number of self-play games to generate")
    p.add_argument("--max-moves", type=int, default=200, help="Maximum moves per game")
    p.add_argument("--simulations", type=int, default=200, help="MCTS simulations per move")
    p.add_argument("--temperature", type=float, default=1.0, help="Move selection temperature")
    p.add_argument("--chunk-size", type=int, default=10_000, help="Rows per parquet file")
    p.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    p.add_argument("--mcts-batch-size", type=int, default=16, help="MCTS model evaluation batch size")
    p.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    p.add_argument("--until-interrupt", action="store_true", help="Continuously generate and train until Ctrl+C")
    p.add_argument("--games-per-iter", type=int, default=50, help="Games to generate per loop iteration (when --until-interrupt)")
    p.add_argument("--epochs-per-iter", type=int, default=1, help="Training epochs per loop iteration (when --until-interrupt)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model: MixtureOfExperts = create_mixture_of_experts(device=device)

    cfg = SelfPlayConfig(
        num_games=int(args.num_games),
        max_moves_per_game=int(args.max_moves),
        simulations_per_move=int(args.simulations),
        temperature=float(args.temperature),
        chunk_size=int(args.chunk_size),
        output_dir=Path(args.output_data),
        mcts_batch_size=int(args.mcts_batch_size),
        device=device,
    )

    if args.until_interrupt:
        print("Starting continuous self-play training. Press Ctrl+C to stop.")
        iteration = 0
        try:
            while True:
                iteration += 1
                # Adjust per-iteration generation count
                iter_cfg = SelfPlayConfig(
                    num_games=int(args.games_per_iter),
                    max_moves_per_game=int(args.max_moves),
                    simulations_per_move=int(args.simulations),
                    temperature=float(args.temperature),
                    chunk_size=int(args.chunk_size),
                    output_dir=Path(args.output_data),
                    mcts_batch_size=int(args.mcts_batch_size),
                    device=device,
                )
                last_idx = _compute_last_batch_index(iter_cfg.output_dir)
                print(f"[Iter {iteration}] Generating {iter_cfg.num_games} self-play games (starting batch {last_idx + 1:06d})...")
                files = generate_self_play_data(model, iter_cfg, last_batch_index=last_idx)
                print(f"[Iter {iteration}] Wrote {len(files)} parquet files to {iter_cfg.output_dir}")

                print(f"[Iter {iteration}] Training for {int(args.epochs_per_iter)} epoch(s)...")
                model = train_from_selfplay(
                    Path(args.output_data),
                    Path(args.output_models),
                    num_epochs=int(args.epochs_per_iter),
                    batch_size=int(args.batch_size),
                    device=device,
                    model=model,
                )
                # Save rolling latest
                latest_path = Path(args.output_models) / "model_latest.pt"
                torch.save(model.state_dict(), latest_path)
                print(f"[Iter {iteration}] Saved latest model: {latest_path}")
        except KeyboardInterrupt:
            interrupted_path = Path(args.output_models) / "model_interrupted.pt"
            os.makedirs(Path(args.output_models), exist_ok=True)
            torch.save(model.state_dict(), interrupted_path)
            print(f"Interrupted. Saved model to: {interrupted_path}")
    else:
        print("Generating self-play data...")
        files = generate_self_play_data(model, cfg)
        print(f"Wrote {len(files)} parquet files to {cfg.output_dir}")

        print("Training from self-play data...")
        _ = train_from_selfplay(
            Path(args.output_data),
            Path(args.output_models),
            num_epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            device=device,
            model=model,
        )


if __name__ == "__main__":
    main()


