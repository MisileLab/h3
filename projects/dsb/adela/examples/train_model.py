"""Example script for training the Adela chess engine."""

import os
import math
import numpy as np
from pathlib import Path
import torch
import polars as pl
from torch.utils.data import DataLoader, random_split

from adela.training.pipeline import (
    create_mixture_of_experts,
    ChessDataset,
    PGNProcessor,
    ParquetProcessor,
    SelfPlayGenerator,
    Trainer
)
from adela.data.hf_dataset import download_split


def train_from_pgn(
    pgn_path: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 256,
    validation_split: float = 0.1,
    min_elo: int = 2000,
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
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
    min_elo: int = 2000,
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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
    min_elo: int = 2000,
    early_stop_patience: int = 3,
    early_stop_min_delta: float = 0.0,
    device: str | None = None
) -> None:
    """Train the model from Parquet data (prioritizes local over HuggingFace).

    Args:
        parquet_path: Path to local directory with split folders or HuggingFace dataset ID.
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

    # Always prioritize local path first, even if it contains "/"
    p = Path(parquet_path)
    root_candidates = [p, p / "data"]
    found_split_root = None
    
    for root in root_candidates:
        if (root / "train").exists() and (root / "validation").exists() and (root / "test").exists():
            found_split_root = root
            break
    
    if found_split_root is not None:
        # Use local data
        print(f"Using local data splits from: {found_split_root}")
        train_data = _process_dir(found_split_root / "train")
        val_data = _process_dir(found_split_root / "validation")
        test_data = _process_dir(found_split_root / "test")
    elif "/" in parquet_path and not p.exists():
        # Fallback to Hugging Face repo if local doesn't exist
        print(f"Local path not found. Detected HF dataset repo: {parquet_path}. Downloading predefined splits...")
        tr_root = download_split(parquet_path, split="train")
        va_root = download_split(parquet_path, split="validation")
        te_root = download_split(parquet_path, split="test")
        train_data = _process_dir(tr_root / "data" / "train")
        val_data = _process_dir(va_root / "data" / "validation")
        test_data = _process_dir(te_root / "data" / "test")
    else:
        raise ValueError(
            f"Expected split folders under '{p}' or '{p}/data' (train/validation/test). If using HuggingFace dataset, ensure the repo exists and is accessible."
        )

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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


def train_from_self_play(
    output_dir: str,
    num_games: int = 1000,
    num_epochs: int = 100,
    batch_size: int = 256,
    validation_split: float = 0.1,
    mcts_simulations: int = 100,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 0.0,
    device: str | None = None
) -> None:
    """Train the model from self-play.

    Args:
        output_dir: Directory to save the model.
        num_games: Number of self-play games to generate.
        num_epochs: Number of epochs to train for.
        batch_size: Batch size.
        validation_split: Fraction of data to use for validation.
        mcts_simulations: Number of MCTS simulations per move.
        device: Device to train on.
    """
    print(f"Training from self-play ({num_games} games)")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    model = create_mixture_of_experts(device=device)
    
    # Generate self-play data
    print("Generating self-play data...")
    generator = SelfPlayGenerator(
        model=model,
        num_games=num_games,
        mcts_simulations=mcts_simulations
    )
    positions, policies, values = generator.generate_games()
    
    print(f"Generated {len(positions)} positions")
    # Save self-play dataset for reuse (NPZ + Parquet)
    save_dir = Path(output_dir) / "self_play_data"
    save_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"sp_{num_games}g_{mcts_simulations}s"

    # NPZ (compressed)
    np.savez_compressed(
        save_dir / f"{base_name}.npz",
        positions=np.array(positions, dtype=object),
        policies=np.stack(policies).astype(np.float32),
        values=np.array(values, dtype=np.float32),
    )
    print(f"Saved NPZ: {(save_dir / f'{base_name}.npz').as_posix()}")

    # Parquet (fen, value, policy[list[float]])
    df = pl.DataFrame(
        {
            "fen": positions,
            "value": values,
            "policy": [p.tolist() for p in policies],
        }
    )
    parquet_path = save_dir / f"{base_name}.parquet"
    df.write_parquet(parquet_path.as_posix())
    print(f"Saved Parquet: {parquet_path.as_posix()}")
    
    # Create dataset
    dataset = ChessDataset(positions, policies, values)
    
    # Split into training and validation sets
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
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
    
    # Train from self-play
    train_from_self_play(
        output_dir="models/self_play",
        num_games=10,  # Small number for example
        num_epochs=3,
        batch_size=64,
        mcts_simulations=50  # Low number for faster training
    )
