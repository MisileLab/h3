"""Hugging Face dataset utilities for Adela.

Provides helpers to split local Parquet files into train/validation/test
and upload them to a Hugging Face dataset repository, and to download
split data locally for training.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from collections.abc import Iterable

from huggingface_hub import create_repo, upload_large_folder, snapshot_download


def _collect_parquet_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")
    return sorted(folder.glob("*.parquet"))


def _ensure_dataset_readme(tmp_dir: Path, repo_id: str) -> Path:
    readme = tmp_dir / "README.md"
    if not readme.exists():
        _ = readme.write_text(
            f"""---
license: cc0-1.0
language:
- en
pretty_name: adela dataset
---
# {repo_id}

Dataset uploaded by Adela tools.

Splits are under `data/train`, `data/validation`, `data/test`.
Files are Parquet with columns similar to Lichess exports plus `parsed_moves`/`num_moves`.
""",
            encoding="utf-8",
        )
    return readme


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

    # Prepare a temp local structure to upload as folder
    tmp_root = Path(".hf_upload_tmp")
    if tmp_root.exists():
        # Clean previous temp if left behind
        for p in tmp_root.rglob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass
        try:
            for d in sorted([p for p in tmp_root.rglob("*") if p.is_dir()], reverse=True):
                d.rmdir()
            tmp_root.rmdir()
        except Exception:
            pass
    (tmp_root / "data" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_root / "data" / "validation").mkdir(parents=True, exist_ok=True)
    (tmp_root / "data" / "test").mkdir(parents=True, exist_ok=True)

    def _link_or_copy(fs: Iterable[Path], dst: Path) -> None:
        for f in fs:
            target = dst / f.name
            try:
                # Hardlink for speed if possible
                os.link(f, target)
            except OSError:
                # Fallback to copy
                target.write_bytes(f.read_bytes())

    _link_or_copy(train_files, tmp_root / "data" / "train")
    _link_or_copy(val_files, tmp_root / "data" / "validation")
    _link_or_copy(test_files, tmp_root / "data" / "test")

    _ = _ensure_dataset_readme(tmp_root, repo_id)

    # Create repo if needed
    _ = create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    # Upload the folder structure (optimized for large datasets)
    _ = upload_large_folder(
        folder_path=str(tmp_root),
        repo_id=repo_id,
        repo_type="dataset",
    )


def download_split(repo_id: str, split: str) -> Path:
    """Download a split folder from a HF dataset repo and return local path.

    Args:
        repo_id: e.g. "misilelab/adela-dataset".
        split: one of {"train", "validation", "test"}.

    Returns:
        Local path to the downloaded snapshot root containing data/{split}/.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: 'train', 'validation', 'test'")

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"data/{split}/*.parquet"],
        local_dir=".hf_datasets_cache",
        local_dir_use_symlinks=False,
    )
    return Path(local_dir)


