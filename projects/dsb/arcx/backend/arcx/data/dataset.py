"""PyTorch Dataset for training

Loads decision logs from Parquet and provides data for training.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
import torch
from torch.utils.data import Dataset

from arcx.data.schema import unflatten_latent_sequence
from arcx.config import config

logger = logging.getLogger(__name__)


class DecisionDataset(Dataset):
    """
    PyTorch Dataset for EV model training.

    Loads decision logs from Parquet files and provides:
    - z_seq: [L, D] latent sequences
    - action: 0 or 1 (stay/extract)
    - target_return: final loot value

    Each sample corresponds to one decision point.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        sequence_length: int = 32,
        latent_dim: int = 512,
        filter_success: Optional[bool] = None,
    ):
        """
        Args:
            data_dir: Directory containing Parquet files
            sequence_length: Expected sequence length L
            latent_dim: Expected latent dimension D
            filter_success: If True, only successful runs; if False, only failed runs; if None, all
        """
        self.data_dir = data_dir or config.data.log_dir
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.filter_success = filter_success

        # Load data
        self.df = self._load_data()

        logger.info(
            f"DecisionDataset loaded: {len(self.df)} samples, "
            f"seq_len={sequence_length}, latent_dim={latent_dim}"
        )

    def _load_data(self) -> pl.DataFrame:
        """Load all Parquet files from data directory"""
        parquet_files = list(self.data_dir.glob("decisions_*.parquet"))

        if not parquet_files:
            logger.warning(f"No Parquet files found in {self.data_dir}")
            raise FileNotFoundError(f"No data files in {self.data_dir}")

        logger.info(f"Loading {len(parquet_files)} Parquet files...")

        # Read all files
        dfs = [pl.read_parquet(f) for f in parquet_files]
        df = pl.concat(dfs)

        logger.info(f"Loaded {len(df)} total decisions")

        # Filter by success if requested
        if self.filter_success is not None:
            df = df.filter(pl.col("success") == self.filter_success)
            logger.info(f"After success filter: {len(df)} samples")

        # Add target_return column (for now, just use final_loot_value)
        # In future, could subtract time penalty here
        df = df.with_columns(
            pl.col("final_loot_value").alias("target_return")
        )

        return df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            z_seq: [L, D] latent sequence
            action: [1] action index (0=stay, 1=extract)
            target_return: [1] target return value
        """
        row = self.df[idx]

        # Extract data
        z_flat = row["z_seq"][0]  # List of floats
        action_str = row["action"][0]
        target_return = row["target_return"][0]

        # Unflatten latent sequence
        z_list = unflatten_latent_sequence(z_flat, self.sequence_length, self.latent_dim)
        z_seq = torch.tensor(z_list, dtype=torch.float32)  # [L, D]

        # Convert action to index
        action_idx = 0 if action_str == "stay" else 1
        action = torch.tensor([action_idx], dtype=torch.long)

        # Target return
        target = torch.tensor([target_return], dtype=torch.float32)

        return z_seq, action, target

    def get_stats(self) -> dict:
        """Get dataset statistics"""
        return {
            "total_samples": len(self.df),
            "unique_runs": self.df["run_id"].n_unique(),
            "action_distribution": self.df.groupby("action").count().to_dict(),
            "mean_target_return": self.df["target_return"].mean(),
            "std_target_return": self.df["target_return"].std(),
        }


def create_dataloaders(
    data_dir: Optional[Path] = None,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Data directory
        batch_size: Batch size
        train_split: Fraction of data for training
        num_workers: Number of dataloader workers

    Returns:
        (train_loader, val_loader)
    """
    from torch.utils.data import random_split, DataLoader

    # Load full dataset
    dataset = DecisionDataset(data_dir=data_dir)

    # Split
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Created dataloaders: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"batch_size={batch_size}"
    )

    return train_loader, val_loader


def test_dataset():
    """Test dataset with dummy data"""
    import tempfile
    from arcx.data.logger import DataLogger

    print("Testing DecisionDataset...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy data
        logger_inst = DataLogger(log_dir=Path(tmpdir))

        for run_idx in range(3):
            logger_inst.start_run(f"run_{run_idx}", "test_map")

            for decision_idx in range(10):
                z_seq = torch.randn(32, 512)
                logger_inst.log_decision(
                    decision_idx=decision_idx,
                    t_sec=decision_idx * 10.0,
                    action="stay" if decision_idx < 9 else "extract",
                    z_seq=z_seq,
                )

            logger_inst.end_run(
                final_loot_value=1000.0 + run_idx * 500,
                total_time_sec=100.0,
                success=True,
            )

        # Load dataset
        dataset = DecisionDataset(data_dir=Path(tmpdir), sequence_length=32, latent_dim=512)
        print(f"Dataset size: {len(dataset)}")
        assert len(dataset) == 30  # 3 runs * 10 decisions

        # Get a sample
        z_seq, action, target = dataset[0]
        print(f"Sample: z_seq={z_seq.shape}, action={action}, target={target}")
        assert z_seq.shape == (32, 512)
        assert action.shape == (1,)
        assert target.shape == (1,)

        # Get stats
        stats = dataset.get_stats()
        print(f"Stats: {stats}")

    print("✓ DecisionDataset tests passed")


if __name__ == "__main__":
    test_dataset()
