"""PyTorch Dataset implementation for ETH volatility prediction."""

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import (
  RANDOM_SEED,
  TRAIN_RATIO,
  VAL_RATIO,
  TEST_RATIO,
  BATCH_SIZE,
)


class ETHVolatilityDataset(Dataset):
  """PyTorch Dataset for ETH volatility and direction prediction."""

  def __init__(self, windows: list[dict[str, Any]]) -> None:
    """
    Initialize the dataset.
    
    Args:
      windows: List of processed windows with embeddings
    """
    self.windows = windows

  def __len__(self) -> int:
    """Return the number of samples in the dataset."""
    return len(self.windows)

  def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
    """
    Get a sample from the dataset.
    
    Args:
      idx: Index of the sample
      
    Returns:
      Dictionary containing:
        - price_seq: Price sequence tensor [window_size, num_features]
        - news_emb: News embedding tensor [embedding_dim]
        - direction: Direction label tensor [1]
        - volatility: Volatility target tensor [1]
    """
    window = self.windows[idx]
    
    # Convert price sequence to tensor
    price_seq = torch.tensor(window["price_sequence"], dtype=torch.float32)
    
    # Convert news embedding to tensor
    news_emb = torch.tensor(window["news_embedding"], dtype=torch.float32)
    
    # Convert labels to tensors
    direction = torch.tensor([window["direction"]], dtype=torch.float32)
    volatility = torch.tensor([window["volatility"]], dtype=torch.float32)
    
    return {
      "price_seq": price_seq,
      "news_emb": news_emb,
      "direction": direction,
      "volatility": volatility,
    }

  def get_feature_shapes(self) -> dict[str, tuple[int, ...]]:
    """Get the shapes of features in the dataset."""
    if not self.windows:
      return {}
    
    sample = self[0]
    return {
      "price_seq": sample["price_seq"].shape,
      "news_emb": sample["news_emb"].shape,
      "direction": sample["direction"].shape,
      "volatility": sample["volatility"].shape,
    }


class DataSplitter:
  """Handles time-based data splitting for training, validation, and testing."""

  def __init__(self, random_seed: int = RANDOM_SEED) -> None:
    """
    Initialize the data splitter.
    
    Args:
      random_seed: Random seed for reproducibility
    """
    self.random_seed = random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

  def time_based_split(self, windows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split data based on time (chronological order).
    
    Args:
      windows: List of processed windows
      
    Returns:
      Tuple of (train_windows, val_windows, test_windows)
    """
    # Sort windows by date to ensure chronological order
    sorted_windows = sorted(windows, key=lambda w: w["date"])
    
    total_samples = len(sorted_windows)
    
    # Calculate split indices
    train_end = int(total_samples * TRAIN_RATIO)
    val_end = int(total_samples * (TRAIN_RATIO + VAL_RATIO))
    
    # Split the data
    train_windows = sorted_windows[:train_end]
    val_windows = sorted_windows[train_end:val_end]
    test_windows = sorted_windows[val_end:]
    
    return train_windows, val_windows, test_windows

  def random_split(self, windows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split data randomly (for comparison/debugging).
    
    Args:
      windows: List of processed windows
      
    Returns:
      Tuple of (train_windows, val_windows, test_windows)
    """
    # Shuffle the data
    shuffled_windows = windows.copy()
    random.shuffle(shuffled_windows)
    
    total_samples = len(shuffled_windows)
    
    # Calculate split indices
    train_end = int(total_samples * TRAIN_RATIO)
    val_end = int(total_samples * (TRAIN_RATIO + VAL_RATIO))
    
    # Split the data
    train_windows = shuffled_windows[:train_end]
    val_windows = shuffled_windows[train_end:val_end]
    test_windows = shuffled_windows[val_end:]
    
    return train_windows, val_windows, test_windows

  def get_split_stats(self, train_windows: list[dict[str, Any]], val_windows: list[dict[str, Any]], test_windows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Get statistics about the data split.
    
    Args:
      train_windows: Training windows
      val_windows: Validation windows
      test_windows: Test windows
      
    Returns:
      Dictionary with split statistics
    """
    total_samples = len(train_windows) + len(val_windows) + len(test_windows)
    
    def get_direction_stats(windows: list[dict[str, Any]]) -> dict[str, Any]:
      if not windows:
        return {"up": 0, "down": 0, "up_ratio": 0.0}
      
      directions = [w["direction"] for w in windows]
      up_count = sum(directions)
      down_count = len(directions) - up_count
      
      return {
        "up": up_count,
        "down": down_count,
        "up_ratio": up_count / len(directions),
      }
    
    def get_volatility_stats(windows: list[dict[str, Any]]) -> dict[str, Any]:
      if not windows:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
      
      volatilities = [w["volatility"] for w in windows]
      
      return {
        "mean": np.mean(volatilities),
        "std": np.std(volatilities),
        "min": np.min(volatilities),
        "max": np.max(volatilities),
      }
    
    def get_date_range(windows: list[dict[str, Any]]) -> dict[str, Any]:
      if not windows:
        return {"start": None, "end": None}
      
      dates = [w["date"] for w in windows]
      return {
        "start": min(dates),
        "end": max(dates),
      }
    
    stats = {
      "total_samples": total_samples,
      "train": {
        "count": len(train_windows),
        "ratio": len(train_windows) / total_samples if total_samples > 0 else 0,
        "direction_stats": get_direction_stats(train_windows),
        "volatility_stats": get_volatility_stats(train_windows),
        "date_range": get_date_range(train_windows),
      },
      "val": {
        "count": len(val_windows),
        "ratio": len(val_windows) / total_samples if total_samples > 0 else 0,
        "direction_stats": get_direction_stats(val_windows),
        "volatility_stats": get_volatility_stats(val_windows),
        "date_range": get_date_range(val_windows),
      },
      "test": {
        "count": len(test_windows),
        "ratio": len(test_windows) / total_samples if total_samples > 0 else 0,
        "direction_stats": get_direction_stats(test_windows),
        "volatility_stats": get_volatility_stats(test_windows),
        "date_range": get_date_range(test_windows),
      },
    }
    
    return stats


def create_data_loaders(
  train_windows: list[dict[str, Any]], 
  val_windows: list[dict[str, Any]], 
  test_windows: list[dict[str, Any]],
  batch_size: int = BATCH_SIZE,
  num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
  """
  Create PyTorch DataLoaders for training, validation, and testing.
  
  Args:
    train_windows: Training windows
    val_windows: Validation windows  
    test_windows: Test windows
    batch_size: Batch size for data loaders
    num_workers: Number of worker processes for data loading
    
  Returns:
    Tuple of (train_loader, val_loader, test_loader)
  """
  # Create datasets
  train_dataset = ETHVolatilityDataset(train_windows)
  val_dataset = ETHVolatilityDataset(val_windows)
  test_dataset = ETHVolatilityDataset(test_windows)
  
  # Create data loaders
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
  
  test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
  )
  
  return train_loader, val_loader, test_loader


def main() -> None:
  """Test the dataset implementation."""
  from .data_loader import CoinDeskDataLoader
  from .preprocessing import DataPreprocessor
  from .embedder import NewsEmbedder
  
  # Load and preprocess data
  loader = CoinDeskDataLoader()
  price_df, news_df = loader.fetch_all_data()
  
  preprocessor = DataPreprocessor()
  windows = preprocessor.process_data(price_df, news_df)
  
  # Add embeddings (using dummy embeddings for testing)
  print("Adding dummy embeddings for testing...")
  for window in windows:
    # Use dummy embeddings for testing
    window["news_embedding"] = np.random.normal(0, 1, 1536).astype(np.float32)
  
  # Split data
  splitter = DataSplitter()
  train_windows, val_windows, test_windows = splitter.time_based_split(windows)
  
  # Get split statistics
  stats = splitter.get_split_stats(train_windows, val_windows, test_windows)
  
  print("Data Split Statistics:")
  for split_name, split_stats in stats.items():
    if split_name == "total_samples":
      print(f"  {split_name}: {split_stats}")
    else:
      print(f"  {split_name}:")
      for key, value in split_stats.items():
        print(f"    {key}: {value}")
  
  # Create datasets and data loaders
  train_loader, val_loader, test_loader = create_data_loaders(
    train_windows, val_windows, test_windows, batch_size=4
  )
  
  print(f"\nDataLoader info:")
  print(f"  Train batches: {len(train_loader)}")
  print(f"  Val batches: {len(val_loader)}")
  print(f"  Test batches: {len(test_loader)}")
  
  # Test a batch
  if len(train_loader) > 0:
    batch = next(iter(train_loader))
    print(f"\nSample batch shapes:")
    for key, tensor in batch.items():
      print(f"  {key}: {tensor.shape}")
  
  # Test dataset feature shapes
  if train_windows:
    train_dataset = ETHVolatilityDataset(train_windows)
    shapes = train_dataset.get_feature_shapes()
    print(f"\nFeature shapes:")
    for key, shape in shapes.items():
      print(f"  {key}: {shape}")


if __name__ == "__main__":
  main()