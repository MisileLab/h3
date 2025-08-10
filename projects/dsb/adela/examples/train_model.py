"""Example script for training the Adela chess engine."""

import os
import math
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from adela.training.pipeline import (
    create_mixture_of_experts,
    ChessDataset,
    PGNProcessor,
    ParquetProcessor,
    StreamingPGNDataset,
    StreamingParquetDataset,
    Trainer,
    collate_training_batch,
)


def train_from_pgn(
    pgn_path: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 256,
    validation_split: float = 0.1,
    min_elo: int = 1000,
    early_stop_patience: int = 3,
    early_stop_min_delta: float = 0.0,
    device: str | None = None
) -> None:
    """Train the model from PGN data.

    Args:
        pgn_path: Path to the PGN file.
        output_dir: Directory to save the model.
        num_epochs: Number of epochs to train for.
        batch_size: Batch size.
        validation_split: Fraction of data to use for validation.
        min_elo: Minimum Elo rating for games to include.
        device: Device to train on.
    """
    print(f"Training from PGN: {pgn_path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    model = create_mixture_of_experts(device=device)
    
    # Process PGN data
    print("Processing PGN data...")
    processor = PGNProcessor(min_elo=min_elo)
    positions, policies, values = processor.process_pgn_file(pgn_path)
    
    print(f"Processed {len(positions)} positions")
    
    # Create dataset
    dataset = ChessDataset(positions, policies, values)
    
    # Split into training and validation sets
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_training_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_training_batch)
    
    # Create trainer
    trainer = Trainer(model, device=device)
    
    # Train the model
    print("Training model...")
    best_val_loss = math.inf
    epochs_no_improve = 0
    best_ckpt_path = None
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        print(f"Training loss: {train_metrics['loss']:.4f}")
        print(f"Training policy loss: {train_metrics['policy_loss']:.4f}")
        print(f"Training value loss: {train_metrics['value_loss']:.4f}")
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        print(f"Validation loss: {val_metrics['val_loss']:.4f}")
        print(f"Validation policy loss: {val_metrics['val_policy_loss']:.4f}")
        print(f"Validation value loss: {val_metrics['val_value_loss']:.4f}")
        
        # Early stopping
        val_loss = float(val_metrics["val_loss"])
        if best_val_loss - val_loss > early_stop_min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_ckpt_path = os.path.join(output_dir, "model_best.pt")
            trainer.save_model(best_ckpt_path)
            print(f"Improved val_loss -> saved best: {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{early_stop_patience})")
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break
    
    # Save final model
    final_model_path = os.path.join(output_dir, "model_final.pt")
    trainer.save_model(final_model_path)
    print(f"Saved final model: {final_model_path}")


def train_from_local_data(
    data_path: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 256,
    min_elo: int = 1000,
    early_stop_patience: int = 3,
    early_stop_min_delta: float = 0.0,
    device: str | None = None
) -> None:
    """Train the model from local Parquet data with predefined splits.

    Args:
        data_path: Path to directory containing train/validation/test folders with parquet files.
        output_dir: Directory to save the model.
        num_epochs: Number of epochs to train for.
        batch_size: Batch size.
        min_elo: Minimum Elo rating for games to include.
        early_stop_patience: Early stopping patience.
        early_stop_min_delta: Minimum improvement for early stopping.
        device: Device to train on.
    """
    print(f"Training from local data: {data_path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model = create_mixture_of_experts(device=device)

    processor = ParquetProcessor(min_elo=min_elo)

    # Look for split folders in the data path
    p = Path(data_path)
    root_candidates = [p, p / "data"]
    found_split_root = None
    
    for root in root_candidates:
        if (root / "train").exists() and (root / "validation").exists() and (root / "test").exists():
            found_split_root = root
            break
    
    if found_split_root is None:
        raise ValueError(
            f"Expected split folders under '{p}' or '{p}/data' (train/validation/test). Please create these folders and place your parquet files inside them."
        )

    print(f"Found data splits in: {found_split_root}")

    def _process_dir(dir_path: Path) -> tuple[list[str], list[np.ndarray], list[float]]:
        return processor.process_parquet(str(dir_path))

    train_data = _process_dir(found_split_root / "train")
    val_data = _process_dir(found_split_root / "validation")
    test_data = _process_dir(found_split_root / "test")

    # Sanity checks
    if len(train_data[0]) == 0:
        raise ValueError("Train split contains zero positions after processing.")
    if len(val_data[0]) == 0:
        raise ValueError("Validation split contains zero positions after processing.")
    if len(test_data[0]) == 0:
        raise ValueError("Test split contains zero positions after processing.")

    print(f"Train positions: {len(train_data[0])}")
    print(f"Validation positions: {len(val_data[0])}")
    print(f"Test positions: {len(test_data[0])}")

    # Build datasets/loaders from explicit splits
    train_dataset = ChessDataset(*train_data)
    val_dataset = ChessDataset(*val_data)
    test_dataset = ChessDataset(*test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_training_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_training_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_training_batch)

    trainer = Trainer(model, device=device)

    print("Training model...")
    best_val_loss = math.inf
    epochs_no_improve = 0
    best_ckpt_path = None
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_metrics = trainer.train_epoch(train_loader)
        print(f"Training loss: {train_metrics['loss']:.4f}")
        print(f"Training policy loss: {train_metrics['policy_loss']:.4f}")
        print(f"Training value loss: {train_metrics['value_loss']:.4f}")

        val_metrics = trainer.validate(val_loader)
        print(f"Validation loss: {val_metrics['val_loss']:.4f}")
        print(f"Validation policy loss: {val_metrics['val_policy_loss']:.4f}")
        print(f"Validation value loss: {val_metrics['val_value_loss']:.4f}")
        
        # Early stopping
        val_loss = float(val_metrics["val_loss"])
        if best_val_loss - val_loss > early_stop_min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_ckpt_path = os.path.join(output_dir, "model_best.pt")
            trainer.save_model(best_ckpt_path)
            print(f"Improved val_loss -> saved best: {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{early_stop_patience})")
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break

    final_model_path = os.path.join(output_dir, "model_final.pt")
    trainer.save_model(final_model_path)
    print(f"Saved final model: {final_model_path}")

    # Test evaluation
    print("Evaluating on test split...")
    test_metrics = trainer.validate(test_loader)
    print(f"Test loss: {test_metrics['val_loss']:.4f}")
    print(f"Test policy loss: {test_metrics['val_policy_loss']:.4f}")
    print(f"Test value loss: {test_metrics['val_value_loss']:.4f}")


def train_from_parquet(
    parquet_path: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 256,
    min_elo: int = 1000,
    early_stop_patience: int = 3,
    early_stop_min_delta: float = 0.0,
    device: str | None = None
) -> None:
    """Train the model from Parquet data using local splits only.

    Args:
        parquet_path: Path to local directory with split folders.
        output_dir: Directory to save the model.
        num_epochs: Number of epochs to train for.
        batch_size: Batch size.
        min_elo: Minimum Elo rating for games to include.
        early_stop_patience: Early stopping patience.
        early_stop_min_delta: Minimum improvement for early stopping.
        device: Device to train on.
    """
    print(f"Training from Parquet: {parquet_path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model = create_mixture_of_experts(device=device)

    processor = ParquetProcessor(min_elo=min_elo)

    # Strictly require predefined splits (train/validation/test) to exist
    train_data: tuple[list[str], list[np.ndarray], list[float]]
    val_data: tuple[list[str], list[np.ndarray], list[float]]
    test_data: tuple[list[str], list[np.ndarray], list[float]]

    def _process_dir(dir_path: Path) -> tuple[list[str], list[np.ndarray], list[float]]:
        return processor.process_parquet(str(dir_path))

    # Always require local path with predefined splits
    p = Path(parquet_path)
    root_candidates = [p, p / "data"]
    found_split_root = None
    
    for root in root_candidates:
        if (root / "train").exists() and (root / "validation").exists() and (root / "test").exists():
            found_split_root = root
            break
    
    if found_split_root is None:
        raise ValueError(
            f"Expected split folders under '{p}' or '{p}/data' (train/validation/test). Please create these folders and place your parquet files inside them."
        )

    print(f"Using local data splits from: {found_split_root}")
    train_data = _process_dir(found_split_root / "train")
    val_data = _process_dir(found_split_root / "validation")
    test_data = _process_dir(found_split_root / "test")

    # Sanity checks
    if len(train_data[0]) == 0:
        raise ValueError("Train split contains zero positions after processing.")
    if len(val_data[0]) == 0:
        raise ValueError("Validation split contains zero positions after processing.")
    if len(test_data[0]) == 0:
        raise ValueError("Test split contains zero positions after processing.")

    print(f"Train positions: {len(train_data[0])}")
    print(f"Validation positions: {len(val_data[0])}")
    print(f"Test positions: {len(test_data[0])}")

    # Build datasets/loaders from explicit splits only
    train_dataset = ChessDataset(*train_data)
    val_dataset = ChessDataset(*val_data)
    test_dataset = ChessDataset(*test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_training_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_training_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_training_batch)

    trainer = Trainer(model, device=device)

    print("Training model...")
    best_val_loss = math.inf
    epochs_no_improve = 0
    best_ckpt_path = None
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_metrics = trainer.train_epoch(train_loader)
        print(f"Training loss: {train_metrics['loss']:.4f}")
        print(f"Training policy loss: {train_metrics['policy_loss']:.4f}")
        print(f"Training value loss: {train_metrics['value_loss']:.4f}")

        val_metrics = trainer.validate(val_loader)
        print(f"Validation loss: {val_metrics['val_loss']:.4f}")
        print(f"Validation policy loss: {val_metrics['val_policy_loss']:.4f}")
        print(f"Validation value loss: {val_metrics['val_value_loss']:.4f}")
        
        # Early stopping
        val_loss = float(val_metrics["val_loss"])
        if best_val_loss - val_loss > early_stop_min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_ckpt_path = os.path.join(output_dir, "model_best.pt")
            trainer.save_model(best_ckpt_path)
            print(f"Improved val_loss -> saved best: {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{early_stop_patience})")
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break

    final_model_path = os.path.join(output_dir, "model_final.pt")
    trainer.save_model(final_model_path)
    print(f"Saved final model: {final_model_path}")

    # Test evaluation
    print("Evaluating on test split...")
    test_metrics = trainer.validate(test_loader)
    print(f"Test loss: {test_metrics['val_loss']:.4f}")
    print(f"Test policy loss: {test_metrics['val_policy_loss']:.4f}")
    print(f"Test value loss: {test_metrics['val_value_loss']:.4f}")


def train_streaming_from_parquet(
    data_path: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 128,
    min_elo: int = 1000,
    max_positions_per_game: int = 30,
    chunk_rows: int = 10_000,
    device: str | None = None,
    early_stop_patience: int = 3,
    early_stop_min_delta: float = 0.0,
    val_data_path: str | None = None,
) -> None:
    """Train using a streaming IterableDataset over Parquet data to minimize RAM usage.

    Args:
      data_path: Path to a single parquet file or a directory of parquet files.
      output_dir: Directory to save checkpoints.
      num_epochs: Number of epochs to iterate over the stream.
      batch_size: Per-iteration batch size; reduce for low GPU RAM.
      min_elo: Elo threshold filter.
      max_positions_per_game: Cap positions extracted per game.
      chunk_rows: How many parquet rows to read per chunk when streaming.
      device: Device string.
    """
    print(f"Streaming training from Parquet at: {data_path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model = create_mixture_of_experts(device=device)

    # If data_path contains split subfolders, use them; otherwise treat data_path as the training source.
    p = Path(data_path)
    root_candidates = [p, p / "data"]
    found_split_root: Path | None = None
    for root in root_candidates:
      if (root / "train").exists() and (root / "validation").exists() and (root / "test").exists():
        found_split_root = root
        break

    train_loader = None
    val_loader = None
    test_loader = None

    if found_split_root is not None:
      print(f"Found data splits in: {found_split_root}")
      train_stream_ds = StreamingParquetDataset(
        data_path=str(found_split_root / "train"),
        min_elo=min_elo,
        max_positions_per_game=max_positions_per_game,
        chunk_rows=chunk_rows,
        shuffle_files=False,
      )
      val_stream_ds = StreamingParquetDataset(
        data_path=str(found_split_root / "validation"),
        min_elo=min_elo,
        max_positions_per_game=max_positions_per_game,
        chunk_rows=chunk_rows,
        shuffle_files=False,
      )
      test_stream_ds = StreamingParquetDataset(
        data_path=str(found_split_root / "test"),
        min_elo=min_elo,
        max_positions_per_game=max_positions_per_game,
        chunk_rows=chunk_rows,
        shuffle_files=False,
      )

      train_loader = DataLoader(
        train_stream_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=collate_training_batch,
      )
      val_loader = DataLoader(
        val_stream_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=collate_training_batch,
      )
      test_loader = DataLoader(
        test_stream_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=collate_training_batch,
      )
    else:
      stream_ds = StreamingParquetDataset(
        data_path=data_path,
        min_elo=min_elo,
        max_positions_per_game=max_positions_per_game,
        chunk_rows=chunk_rows,
        shuffle_files=False,
      )
      train_loader = DataLoader(
        stream_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=collate_training_batch,
      )
      if val_data_path is not None:
        val_stream_ds = StreamingParquetDataset(
          data_path=val_data_path,
          min_elo=min_elo,
          max_positions_per_game=max_positions_per_game,
          chunk_rows=chunk_rows,
          shuffle_files=False,
        )
        val_loader = DataLoader(
          val_stream_ds,
          batch_size=batch_size,
          shuffle=False,
          num_workers=0,
          pin_memory=(device == "cuda"),
          collate_fn=collate_training_batch,
        )

    trainer = Trainer(model, device=device, batch_size=batch_size)

    print("Training model (streaming)...")
    best_metric = math.inf
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(output_dir, "model_best.pt")

    # val_loader may be None if no validation split or path provided

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_metrics = trainer.train_epoch(train_loader)
        print(f"Training loss: {train_metrics['loss']:.4f}")

        # Validate if a separate validation stream is provided; otherwise use train loss
        current_metric: float
        if val_loader is not None:
          val_metrics = trainer.validate(val_loader)
          current_metric = float(val_metrics["val_loss"])
          print(f"Validation loss: {current_metric:.4f}")
        else:
          current_metric = float(train_metrics["loss"])

        # Early stopping on current_metric
        if best_metric - current_metric > early_stop_min_delta:
          best_metric = current_metric
          epochs_no_improve = 0
          trainer.save_model(best_ckpt_path)
          print(f"Improved metric -> saved best: {best_ckpt_path}")
        else:
          epochs_no_improve += 1
          print(f"No improvement ({epochs_no_improve}/{early_stop_patience})")
          if epochs_no_improve >= early_stop_patience:
            print("Early stopping triggered.")
            break

    final_model_path = os.path.join(output_dir, "model_final.pt")
    trainer.save_model(final_model_path)
    print(f"Saved final model: {final_model_path}")

    # Optional test evaluation if split detected
    if test_loader is not None:
      print("Evaluating on test split...")
      test_metrics = trainer.validate(test_loader)
      print(f"Test loss: {test_metrics['val_loss']:.4f}")
      print(f"Test policy loss: {test_metrics['val_policy_loss']:.4f}")
      print(f"Test value loss: {test_metrics['val_value_loss']:.4f}")


def train_streaming_from_pgn(
    pgn_path: str,
    output_dir: str,
    num_epochs: int = 5,
    batch_size: int = 64,
    min_elo: int = 1000,
    max_positions_per_game: int = 30,
    device: str | None = None,
    early_stop_patience: int = 3,
    early_stop_min_delta: float = 0.0,
    val_pgn_path: str | None = None,
) -> None:
    """Train using a streaming IterableDataset over a PGN file to minimize RAM usage."""
    print(f"Streaming training from PGN: {pgn_path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model = create_mixture_of_experts(device=device)

    stream_ds = StreamingPGNDataset(
        pgn_path=pgn_path,
        min_elo=min_elo,
        max_positions_per_game=max_positions_per_game,
    )

    train_loader = DataLoader(
        stream_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
      pin_memory=(device == "cuda"),
      collate_fn=collate_training_batch,
    )

    trainer = Trainer(model, device=device, batch_size=batch_size)

    print("Training model (streaming)...")
    best_metric = math.inf
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(output_dir, "model_best.pt")

    # Optional validation stream
    val_loader = None
    if val_pgn_path is not None:
      val_stream_ds = StreamingPGNDataset(
        pgn_path=val_pgn_path,
        min_elo=min_elo,
        max_positions_per_game=max_positions_per_game,
      )
      val_loader = DataLoader(
        val_stream_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=collate_training_batch,
      )

    for epoch in range(num_epochs):
      print(f"Epoch {epoch+1}/{num_epochs}")
      train_metrics = trainer.train_epoch(train_loader)
      print(f"Training loss: {train_metrics['loss']:.4f}")

      current_metric: float
      if val_loader is not None:
        val_metrics = trainer.validate(val_loader)
        current_metric = float(val_metrics["val_loss"])
        print(f"Validation loss: {current_metric:.4f}")
      else:
        current_metric = float(train_metrics["loss"])

      if best_metric - current_metric > early_stop_min_delta:
        best_metric = current_metric
        epochs_no_improve = 0
        trainer.save_model(best_ckpt_path)
        print(f"Improved metric -> saved best: {best_ckpt_path}")
      else:
        epochs_no_improve += 1
        print(f"No improvement ({epochs_no_improve}/{early_stop_patience})")
        if epochs_no_improve >= early_stop_patience:
          print("Early stopping triggered.")
          break

    final_model_path = os.path.join(output_dir, "model_final.pt")
    trainer.save_model(final_model_path)
    print(f"Saved final model: {final_model_path}")

def train_from_batch_parquet(
    parquet_files: list[str] | list[Path],
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 256,
    validation_split: float = 0.1,
    min_elo: int = 2000,
    max_positions_per_game: int = 30,
    early_stop_patience: int = 3,
    early_stop_min_delta: float = 0.0,
    device: str | None = None
) -> None:
    """Train the model from a list of parquet files using a simple Dataset.

    The parquet files are fully processed into position-level arrays and fed
    via a standard PyTorch Dataset and DataLoader.
    """
    print(f"Training from {len(parquet_files)} parquet files (simple Dataset)")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model = create_mixture_of_experts(device=device)

    processor = ParquetProcessor(min_elo=min_elo, max_positions_per_game=max_positions_per_game)

    # Split file list for validation
    val_size = max(1, int(validation_split * len(parquet_files)))
    train_files = parquet_files[:-val_size]
    val_files = parquet_files[-val_size:]

    def _process_many(files: list[str] | list[Path]) -> tuple[list[str], list[np.ndarray], list[float]]:
        all_pos: list[str] = []
        all_pol: list[np.ndarray] = []
        all_val: list[float] = []
        for f in files:
            p, pol, v = processor.process_parquet(str(f))
            all_pos.extend(p)
            all_pol.extend(pol)
            all_val.extend(v)
        return all_pos, all_pol, all_val

    print("Processing training parquet files into positions...")
    train_positions, train_policies, train_values = _process_many(train_files)
    print("Processing validation parquet files into positions...")
    val_positions, val_policies, val_values = _process_many(val_files)

    print(f"Train positions: {len(train_positions)}")
    print(f"Validation positions: {len(val_positions)}")

    train_dataset = ChessDataset(train_positions, train_policies, train_values)
    val_dataset = ChessDataset(val_positions, val_policies, val_values)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

    trainer = Trainer(model, device=device)

    print("Training model...")
    best_val_loss = math.inf
    epochs_no_improve = 0
    best_ckpt_path = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_metrics = trainer.train_epoch(train_loader)
        print(f"Training loss: {train_metrics['loss']:.4f}")
        print(f"Training policy loss: {train_metrics['policy_loss']:.4f}")
        print(f"Training value loss: {train_metrics['value_loss']:.4f}")

        val_metrics = trainer.validate(val_loader)
        print(f"Validation loss: {val_metrics['val_loss']:.4f}")
        print(f"Validation policy loss: {val_metrics['val_policy_loss']:.4f}")
        print(f"Validation value loss: {val_metrics['val_value_loss']:.4f}")

        val_loss = float(val_metrics["val_loss"])
        if best_val_loss - val_loss > early_stop_min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_ckpt_path = os.path.join(output_dir, "model_best.pt")
            trainer.save_model(best_ckpt_path)
            print(f"Improved val_loss -> saved best: {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{early_stop_patience})")
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break

    final_model_path = os.path.join(output_dir, "model_final.pt")
    trainer.save_model(final_model_path)
    print(f"Saved final model: {final_model_path}")


# Self-play training has been removed to enforce local-only data usage


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    Path("examples").mkdir(exist_ok=True)
    
    # Example usage
    # Uncomment one of these to run the example
    
    # Train from local data with predefined splits (RECOMMENDED)
    # train_from_local_data(
    #     data_path="data/games",  # Path to folder containing train/validation/test subfolders
    #     output_dir="models/local_data_trained",
    #     num_epochs=5,
    #     batch_size=256,
    #     min_elo=1000
    # )
    
    # Train from PGN data
    # train_from_pgn(
    #     pgn_path="path/to/your/games.pgn",
    #     output_dir="models/pgn_trained",
    #     num_epochs=5,
    #     batch_size=64
    # )
    
    # Train from parquet files using simple Dataset
    # parquet_files = list(Path("data/games").glob("*.parquet"))
    # train_from_batch_parquet(
    #     parquet_files=parquet_files,
    #     output_dir="models/parquet_trained",
    #     num_epochs=5,
    #     batch_size=256,
    #     min_elo=2000
    # )
    
    # Self-play example removed. Use local data examples above.
