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
    BatchParquetDataset,
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


def train_from_parquet(
    parquet_path: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 256,
    min_elo: int = 2000,
    parse_chunk_size: int = 1000,
    early_stop_patience: int = 3,
    early_stop_min_delta: float = 0.0,
    device: str | None = None
) -> None:
    """Train the model from Parquet data.

    Args:
        parquet_path: Path to a .parquet file or directory with parquet files.
        output_dir: Directory to save the model.
        num_epochs: Number of epochs to train for.
        batch_size: Batch size.
        validation_split: Fraction of data to use for validation.
        min_elo: Minimum Elo rating for games to include.
        parse_chunk_size: Chunk size for movetext parsing if needed.
        device: Device to train on.
    """
    print(f"Training from Parquet or HF dataset: {parquet_path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model = create_mixture_of_experts(device=device)

    processor = ParquetProcessor(min_elo=min_elo, parse_chunk_size=parse_chunk_size)

    # Strictly require predefined splits (train/validation/test) to exist
    train_data: tuple[list[str], list[np.ndarray], list[float]]
    val_data: tuple[list[str], list[np.ndarray], list[float]]
    test_data: tuple[list[str], list[np.ndarray], list[float]]

    def _process_dir(dir_path: Path) -> tuple[list[str], list[np.ndarray], list[float]]:
        return processor.process_parquet(str(dir_path))

    if "/" in parquet_path and not Path(parquet_path).exists():
        # Hugging Face repo id: download each split explicitly
        print(f"Detected HF dataset repo: {parquet_path}. Downloading predefined splits (train/validation/test)...")
        tr_root = download_split(parquet_path, split="train")
        va_root = download_split(parquet_path, split="validation")
        te_root = download_split(parquet_path, split="test")
        train_data = _process_dir(tr_root / "data" / "train")
        val_data = _process_dir(va_root / "data" / "validation")
        test_data = _process_dir(te_root / "data" / "test")
    else:
        # Local path: expect data/train|validation|test or train|validation|test
        p = Path(parquet_path)
        root_candidates = [p, p / "data"]
        found_split_root = None
        for root in root_candidates:
            if (root / "train").exists() and (root / "validation").exists() and (root / "test").exists():
                found_split_root = root
                break
        if found_split_root is None:
            raise ValueError(
                f"Expected split folders under '{p}' (train/validation/test or data/train|validation|test)."
            )
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
    parquet_batch_size: int = 1000,
    validation_split: float = 0.1,
    min_elo: int = 2000,
    max_positions_per_game: int = 30,
    early_stop_patience: int = 3,
    early_stop_min_delta: float = 0.0,
    device: str | None = None
) -> None:
    """Train the model using BatchParquetDataset for efficient memory usage.

    Args:
        parquet_files: List of paths to parquet files.
        output_dir: Directory to save the model.
        num_epochs: Number of epochs to train for.
        batch_size: Training batch size.
        parquet_batch_size: Size of batches to load from parquet files.
        validation_split: Fraction of files to use for validation.
        min_elo: Minimum Elo rating for games to include.
        max_positions_per_game: Maximum positions to extract per game.
        early_stop_patience: Early stopping patience.
        early_stop_min_delta: Minimum improvement for early stopping.
        device: Device to train on.
    """
    print(f"Training from {len(parquet_files)} parquet files using BatchParquetDataset")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    model = create_mixture_of_experts(device=device)
    
    # Split files into train/validation
    val_size = max(1, int(validation_split * len(parquet_files)))
    train_files = parquet_files[:-val_size]
    val_files = parquet_files[-val_size:]
    
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    
    # Create datasets using BatchParquetDataset
    train_dataset = BatchParquetDataset(
        file_paths=train_files,
        batch_size=parquet_batch_size,
        min_elo=min_elo,
        max_positions_per_game=max_positions_per_game
    )
    
    val_dataset = BatchParquetDataset(
        file_paths=val_files,
        batch_size=parquet_batch_size,
        min_elo=min_elo,
        max_positions_per_game=max_positions_per_game
    )
    
    print(f"Training games: {len(train_dataset)}")
    print(f"Validation games: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
    
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


def train_from_self_play(
    output_dir: str,
    num_games: int = 100,
    num_epochs: int = 10,
    batch_size: int = 256,
    validation_split: float = 0.1,
    mcts_simulations: int = 100,
    early_stop_patience: int = 3,
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
    
    # Train from PGN data
    # train_from_pgn(
    #     pgn_path="path/to/your/games.pgn",
    #     output_dir="models/pgn_trained",
    #     num_epochs=5,
    #     batch_size=64
    # )
    
    # Train from batch parquet files (most efficient for large datasets)
    # parquet_files = list(Path("data/games").glob("*.parquet"))
    # train_from_batch_parquet(
    #     parquet_files=parquet_files,
    #     output_dir="models/batch_parquet_trained",
    #     num_epochs=5,
    #     batch_size=256,
    #     parquet_batch_size=1000,
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
