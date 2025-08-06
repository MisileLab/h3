"""Training script for ETH volatility and direction prediction model."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error
from safetensors.torch import save_file
from tqdm import tqdm

from .config import (
  RANDOM_SEED,
  LEARNING_RATE,
  NUM_EPOCHS,
  EARLY_STOPPING_PATIENCE,
  ALPHA_VOLATILITY_WEIGHT,
  EXPERIMENTS_DIR,
  MODEL_CHECKPOINT_FILE,
  LOG_LEVEL,
  LOG_FORMAT,
)
from .model import ETHVolatilityModel, CombinedLoss
from .dataset import create_data_loaders, DataSplitter
from .data_loader import CoinDeskDataLoader
from .preprocessing import DataPreprocessor
from .embedder import NewsEmbedder


# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class Trainer:
  """Handles model training and validation."""

  def __init__(
    self, 
    model: ETHVolatilityModel, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float = LEARNING_RATE,
    alpha: float = ALPHA_VOLATILITY_WEIGHT,
  ) -> None:
    """
    Initialize the trainer.
    
    Args:
      model: The model to train
      train_loader: Training data loader
      val_loader: Validation data loader  
      device: Device to train on
      learning_rate: Learning rate for optimizer
      alpha: Weight for volatility loss in combined loss
    """
    self.model = model.to(device)
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.device = device
    
    # Loss function and optimizer
    self.criterion = CombinedLoss(alpha=alpha)
    self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training state
    self.epoch = 0
    self.best_val_loss = float('inf')
    self.epochs_without_improvement = 0
    self.train_history: list[dict[str, float]] = []
    self.val_history: list[dict[str, float]] = []

  def train_epoch(self) -> dict[str, float]:
    """Train for one epoch."""
    self.model.train()
    
    total_losses = {
      "direction_loss": 0.0,
      "volatility_loss": 0.0,
      "combined_loss": 0.0,
    }
    
    all_direction_preds = []
    all_direction_targets = []
    all_volatility_preds = []
    all_volatility_targets = []
    
    num_batches = len(self.train_loader)
    
    with tqdm(self.train_loader, desc=f"Training Epoch {self.epoch + 1}") as pbar:
      for batch in pbar:
        # Move data to device
        price_seq = batch["price_seq"].to(self.device)
        news_emb = batch["news_emb"].to(self.device)
        direction_target = batch["direction"].to(self.device)
        volatility_target = batch["volatility"].to(self.device)
        
        # Forward pass
        predictions = self.model(price_seq, news_emb)
        
        # Compute loss
        targets = {
          "direction": direction_target,
          "volatility": volatility_target,
        }
        losses = self.criterion(predictions, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses["combined_loss"].backward()
        self.optimizer.step()
        
        # Accumulate losses
        for key, loss in losses.items():
          total_losses[key] += loss.item()
        
        # Collect predictions for metrics
        all_direction_preds.extend(predictions["direction"].cpu().detach().numpy())
        all_direction_targets.extend(direction_target.cpu().detach().numpy())
        all_volatility_preds.extend(predictions["volatility"].cpu().detach().numpy())
        all_volatility_targets.extend(volatility_target.cpu().detach().numpy())
        
        # Update progress bar
        pbar.set_postfix({
          "Loss": f"{losses['combined_loss'].item():.4f}",
          "Dir": f"{losses['direction_loss'].item():.4f}",
          "Vol": f"{losses['volatility_loss'].item():.4f}",
        })
    
    # Calculate average losses
    avg_losses = {key: loss / num_batches for key, loss in total_losses.items()}
    
    # Calculate metrics
    metrics = self._calculate_metrics(
      all_direction_preds, all_direction_targets,
      all_volatility_preds, all_volatility_targets
    )
    
    # Combine losses and metrics
    epoch_results = {**avg_losses, **metrics}
    
    return epoch_results

  def validate_epoch(self) -> dict[str, float]:
    """Validate for one epoch."""
    self.model.eval()
    
    total_losses = {
      "direction_loss": 0.0,
      "volatility_loss": 0.0,
      "combined_loss": 0.0,
    }
    
    all_direction_preds = []
    all_direction_targets = []
    all_volatility_preds = []
    all_volatility_targets = []
    
    num_batches = len(self.val_loader)
    
    with torch.no_grad():
      with tqdm(self.val_loader, desc=f"Validation Epoch {self.epoch + 1}") as pbar:
        for batch in pbar:
          # Move data to device
          price_seq = batch["price_seq"].to(self.device)
          news_emb = batch["news_emb"].to(self.device)
          direction_target = batch["direction"].to(self.device)
          volatility_target = batch["volatility"].to(self.device)
          
          # Forward pass
          predictions = self.model(price_seq, news_emb)
          
          # Compute loss
          targets = {
            "direction": direction_target,
            "volatility": volatility_target,
          }
          losses = self.criterion(predictions, targets)
          
          # Accumulate losses
          for key, loss in losses.items():
            total_losses[key] += loss.item()
          
          # Collect predictions for metrics
          all_direction_preds.extend(predictions["direction"].cpu().detach().numpy())
          all_direction_targets.extend(direction_target.cpu().detach().numpy())
          all_volatility_preds.extend(predictions["volatility"].cpu().detach().numpy())
          all_volatility_targets.extend(volatility_target.cpu().detach().numpy())
          
          # Update progress bar
          pbar.set_postfix({
            "Loss": f"{losses['combined_loss'].item():.4f}",
            "Dir": f"{losses['direction_loss'].item():.4f}",
            "Vol": f"{losses['volatility_loss'].item():.4f}",
          })
    
    # Calculate average losses
    avg_losses = {key: loss / num_batches for key, loss in total_losses.items()}
    
    # Calculate metrics
    metrics = self._calculate_metrics(
      all_direction_preds, all_direction_targets,
      all_volatility_preds, all_volatility_targets
    )
    
    # Combine losses and metrics
    epoch_results = {**avg_losses, **metrics}
    
    return epoch_results

  def _calculate_metrics(
    self, 
    direction_preds: list[float], 
    direction_targets: list[float],
    volatility_preds: list[float], 
    volatility_targets: list[float]
  ) -> dict[str, float]:
    """Calculate evaluation metrics."""
    import numpy as np
    
    # Convert to numpy arrays and flatten
    dir_preds = np.array(direction_preds).flatten()
    dir_targets = np.array(direction_targets).flatten()
    vol_preds = np.array(volatility_preds).flatten()
    vol_targets = np.array(volatility_targets).flatten()
    
    # Direction metrics
    dir_binary_preds = (dir_preds > 0.5).astype(int)
    direction_accuracy = accuracy_score(dir_targets, dir_binary_preds)
    
    try:
      direction_auc = roc_auc_score(dir_targets, dir_preds)
    except ValueError:
      # Handle case where all targets are the same class
      direction_auc = 0.5
    
    # Volatility metrics
    volatility_mae = mean_absolute_error(vol_targets, vol_preds)
    volatility_rmse = mean_squared_error(vol_targets, vol_preds, squared=False)
    
    return {
      "direction_accuracy": direction_accuracy,
      "direction_auc": direction_auc,
      "volatility_mae": volatility_mae,
      "volatility_rmse": volatility_rmse,
    }

  def save_checkpoint(self, filepath: Path, is_best: bool = False) -> None:
    """Save model checkpoint."""
    checkpoint = {
      "model_state_dict": self.model.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "scheduler_state_dict": self.scheduler.state_dict(),
      "epoch": self.epoch,
      "best_val_loss": self.best_val_loss,
      "train_history": self.train_history,
      "val_history": self.val_history,
    }
    
    save_file(checkpoint, str(filepath))
    
    if is_best:
      best_filepath = filepath.parent / f"best_{filepath.name}"
      save_file(checkpoint, str(best_filepath))

  def load_checkpoint(self, filepath: Path) -> None:
    """Load model checkpoint."""
    from safetensors.torch import load_file
    
    checkpoint = load_file(str(filepath))
    
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    self.epoch = checkpoint["epoch"]
    self.best_val_loss = checkpoint["best_val_loss"]
    self.train_history = checkpoint["train_history"]
    self.val_history = checkpoint["val_history"]

  def train(self, num_epochs: int = NUM_EPOCHS, patience: int = EARLY_STOPPING_PATIENCE) -> dict[str, Any]:
    """
    Train the model for multiple epochs.
    
    Args:
      num_epochs: Number of epochs to train
      patience: Early stopping patience
      
    Returns:
      Training results dictionary
    """
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Device: {self.device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
      self.epoch = epoch
      
      # Train epoch
      train_results = self.train_epoch()
      self.train_history.append(train_results)
      
      # Validate epoch
      val_results = self.validate_epoch()
      self.val_history.append(val_results)
      
      # Learning rate scheduling
      self.scheduler.step(val_results["combined_loss"])
      
      # Log results
      logger.info(f"Epoch {epoch + 1}/{num_epochs}")
      logger.info(f"  Train - Loss: {train_results['combined_loss']:.4f}, "
                 f"Dir Acc: {train_results['direction_accuracy']:.4f}, "
                 f"Vol RMSE: {train_results['volatility_rmse']:.4f}")
      logger.info(f"  Val   - Loss: {val_results['combined_loss']:.4f}, "
                 f"Dir Acc: {val_results['direction_accuracy']:.4f}, "
                 f"Vol RMSE: {val_results['volatility_rmse']:.4f}")
      
      # Check for improvement
      if val_results["combined_loss"] < self.best_val_loss:
        self.best_val_loss = val_results["combined_loss"]
        self.epochs_without_improvement = 0
        
        # Save best model
        checkpoint_path = EXPERIMENTS_DIR / MODEL_CHECKPOINT_FILE
        self.save_checkpoint(checkpoint_path, is_best=True)
        logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
      else:
        self.epochs_without_improvement += 1
      
      # Early stopping
      if self.epochs_without_improvement >= patience:
        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
        break
    
    end_time = time.time()
    training_time = end_time - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
    
    return {
      "num_epochs_trained": epoch + 1,
      "training_time": training_time,
      "best_val_loss": self.best_val_loss,
      "final_train_results": self.train_history[-1] if self.train_history else {},
      "final_val_results": self.val_history[-1] if self.val_history else {},
      "train_history": self.train_history,
      "val_history": self.val_history,
    }


def main() -> None:
  """Main training pipeline."""
  # Set random seed for reproducibility
  torch.manual_seed(RANDOM_SEED)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
  
  # Set device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logger.info(f"Using device: {device}")
  
  try:
    # Load and preprocess data
    logger.info("Loading data...")
    loader = CoinDeskDataLoader()
    price_df, news_df = loader.fetch_all_data()
    
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    windows = preprocessor.process_data(price_df, news_df)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embedder = NewsEmbedder()
    windows = embedder.process_windows(windows)
    
    # Split data
    logger.info("Splitting data...")
    splitter = DataSplitter()
    train_windows, val_windows, test_windows = splitter.time_based_split(windows)
    
    # Log split statistics
    stats = splitter.get_split_stats(train_windows, val_windows, test_windows)
    logger.info(f"Data split - Train: {stats['train']['count']}, "
               f"Val: {stats['val']['count']}, Test: {stats['test']['count']}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
      train_windows, val_windows, test_windows
    )
    
    # Determine input size
    sample_batch = next(iter(train_loader))
    price_input_size = sample_batch["price_seq"].shape[-1]
    
    # Create model
    logger.info("Creating model...")
    from .model import create_model
    model = create_model(price_input_size)
    
    # Log model info
    model_info = model.get_model_info()
    logger.info(f"Model created with {model_info['total_parameters']:,} parameters")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # Train model
    logger.info("Starting training...")
    training_results = trainer.train()
    
    # Save training results
    results_file = EXPERIMENTS_DIR / "training_results.json"
    with open(results_file, "w") as f:
      # Convert numpy types to Python types for JSON serialization
      def convert_types(obj: Any) -> Any:
        if hasattr(obj, 'item'):
          return obj.item()
        elif isinstance(obj, dict):
          return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
          return [convert_types(v) for v in obj]
        else:
          return obj
      
      json.dump(convert_types(training_results), f, indent=2, default=str)
    
    logger.info(f"Training results saved to {results_file}")
    logger.info("Training pipeline completed successfully!")
    
  except Exception as e:
    logger.error(f"Training failed with error: {e}")
    raise


if __name__ == "__main__":
  main()