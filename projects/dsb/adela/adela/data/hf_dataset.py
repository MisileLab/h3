"""Hugging Face dataset utilities for Adela.

Provides helpers to split local Parquet files into train/validation/test
and upload them to a Hugging Face dataset repository, and to download
split data locally for training.
"""

from __future__ import annotations
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


def _collect_parquet_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")
    return sorted(folder.glob("*.parquet"))



def split_and_upload_parquet(
    source_folder: str | Path,
    repo_id: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    commit_message: str = "Upload dataset with auto splits",
) -> None:
    """Split local Parquet files into train/val/test and upload to HF datasets.

    Splitting is file-level by random shuffling with provided ratios.

    Args:
        source_folder: Directory containing processed Parquet files.
        repo_id: Target dataset repo, e.g. "misilelab/adela-dataset".
        train_ratio: Proportion of files for the train split.
        val_ratio: Proportion for validation split.
        test_ratio: Proportion for test split.
        commit_message: Commit message for the upload.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    src = Path(source_folder)
    files = _collect_parquet_files(src)
    if not files:
        raise ValueError(f"No .parquet files found in {src}")

    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    _ = n - n_train - n_val  # implicit test size

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    # Build a DatasetDict directly from local Parquet files
    data_files = {
        "train": [str(p) for p in train_files],
        "validation": [str(p) for p in val_files],
        "test": [str(p) for p in test_files],
    }

    dsdict = load_dataset("parquet", data_files=data_files)
    # Push to the Hub as a proper datasets repository
    _ = dsdict.push_to_hub(repo_id=repo_id, commit_message=commit_message)


def download_split(repo_id: str, split: str) -> Path:
    """Materialize a split from a Hub dataset to local Parquet files and return root.

    The returned path will contain ``data/{split}/`` with one parquet file at minimum,
    so existing consumers can read from ``root / 'data' / split``.

    Args:
        repo_id: e.g. "misilelab/adela-dataset".
        split: one of {"train", "validation", "test"}.

    Returns:
        Local root path containing ``data/{split}/``.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: 'train', 'validation', 'test'")

    # Load via datasets API to be robust to repository layout
    ds_any = load_dataset(repo_id, split=split)

    root = Path(".hf_datasets_cache") / repo_id.replace("/", "__")
    out_dir = root / "data" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(ds_any, Dataset):
        _ = ds_any.to_parquet(str(out_dir / "data.parquet"))
        return root
    if isinstance(ds_any, DatasetDict):
        subset = ds_any[split]
        _ = subset.to_parquet(str(out_dir / "data.parquet"))
        return root

    raise TypeError(f"Unexpected datasets type: {type(ds_any)}")


