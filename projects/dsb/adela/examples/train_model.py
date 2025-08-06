"""Example script for training the Adela chess engine."""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from adela.training.pipeline import (
    create_mixture_of_experts,
    ChessDataset,
    PGNProcessor,
    SelfPlayGenerator,
    Trainer
)


def train_from_pgn(
    pgn_path: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 256,
    validation_split: float = 0.1,
    min_elo: int = 2000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
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
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
        trainer.save_model(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
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
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
        trainer.save_model(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
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
    
    # Train from self-play
    train_from_self_play(
        output_dir="models/self_play",
        num_games=10,  # Small number for example
        num_epochs=3,
        batch_size=64,
        mcts_simulations=50  # Low number for faster training
    )
