"""Data preprocessing for ETH volatility and direction prediction."""

import numpy as np
import polars as pl
from typing import Any
from datetime import date

from .config import WINDOW_SIZE


class DataPreprocessor:
  """Preprocesses price and news data for model training."""

  def __init__(self) -> None:
    """Initialize the preprocessor."""
    pass

  def compute_returns_and_volatility(self, price_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute daily returns and realized volatility.
    
    Args:
      price_df: DataFrame with OHLCV data
      
    Returns:
      DataFrame with additional columns: direction, daily_return, realized_volatility
    """
    df = price_df.clone()
    
    # Sort by date to ensure proper ordering
    df = df.sort("date")
    
    # Compute daily direction (1 if close > open, 0 otherwise)
    df = df.with_columns([
      (pl.col("close") > pl.col("open")).cast(pl.Int32).alias("direction"),
    ])
    
    # Compute daily returns (close-to-close)
    df = df.with_columns([
      ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1)).alias("daily_return"),
    ])
    
    # Compute intraday returns for volatility calculation
    df = df.with_columns([
      ((pl.col("high") - pl.col("open")) / pl.col("open")).alias("high_return"),
      ((pl.col("low") - pl.col("open")) / pl.col("open")).alias("low_return"),
      ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("close_return"),
    ])
    
    # Compute realized volatility using Garman-Klass estimator
    df = df.with_columns([
      (
        0.5 * (pl.col("high_return") - pl.col("low_return")).pow(2) -
        (2 * np.log(2) - 1) * pl.col("close_return").pow(2)
      ).sqrt().alias("realized_volatility")
    ])
    
    # Fill NaN values in volatility with 0
    df = df.with_columns([
      pl.col("realized_volatility").fill_nan(0.0),
      pl.col("daily_return").fill_nan(0.0),
    ])
    
    # Drop intermediate columns
    df = df.drop(["high_return", "low_return", "close_return"])
    
    return df

  def normalize_price_features(self, price_df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize price features for model input.
    
    Args:
      price_df: DataFrame with price data
      
    Returns:
      DataFrame with normalized features
    """
    df = price_df.clone()
    
    # Compute percentage changes for OHLC
    for col in ["open", "high", "low", "close"]:
      df = df.with_columns([
        ((pl.col(col) - pl.col(col).shift(1)) / pl.col(col).shift(1)).alias(f"{col}_pct_change")
      ])
    
    # Compute volume percentage change
    df = df.with_columns([
      ((pl.col("volume") - pl.col("volume").shift(1)) / pl.col("volume").shift(1)).alias("volume_pct_change")
    ])
    
    # Fill NaN values with 0
    pct_change_cols = [f"{col}_pct_change" for col in ["open", "high", "low", "close"]] + ["volume_pct_change"]
    for col in pct_change_cols:
      df = df.with_columns([
        pl.col(col).fill_nan(0.0)
      ])
    
    return df

  def align_data_by_date(self, price_df: pl.DataFrame, news_df: pl.DataFrame) -> pl.DataFrame:
    """
    Align price and news data by date.
    
    Args:
      price_df: DataFrame with price data
      news_df: DataFrame with news data
      
    Returns:
      DataFrame with aligned data
    """
    # Join on date
    aligned_df = price_df.join(news_df, on="date", how="left")
    
    # Fill missing news with empty strings
    aligned_df = aligned_df.with_columns([
      pl.col("headline").fill_null(""),
      pl.col("content").fill_null(""),
      pl.col("url").fill_null(""),
    ])
    
    return aligned_df

  def create_sliding_windows(self, aligned_df: pl.DataFrame) -> list[dict[str, Any]]:
    """
    Create sliding windows for model input.
    
    Args:
      aligned_df: DataFrame with aligned price and news data
      
    Returns:
      List of dictionaries with window data
    """
    # Sort by date
    df = aligned_df.sort("date")
    
    # Feature columns for price sequences
    price_feature_cols = [
      "open_pct_change", "high_pct_change", "low_pct_change", 
      "close_pct_change", "volume_pct_change"
    ]
    
    windows = []
    
    # Create sliding windows
    for i in range(WINDOW_SIZE, len(df)):
      # Get window data
      window_data = df[i - WINDOW_SIZE:i]
      current_row = df[i]
      
      # Extract price sequence (WINDOW_SIZE x num_features)
      price_sequence = []
      for _, row in enumerate(window_data.iter_rows(named=True)):
        price_features = [row[col] for col in price_feature_cols]
        price_sequence.append(price_features)
      
      # Current day's news embedding placeholder (will be filled by embedder)
      current_news = {
        "headline": current_row["headline"][0] if len(current_row) > 0 else "",
        "content": current_row["content"][0] if len(current_row) > 0 else "",
      }
      
      # Labels
      direction = int(current_row["direction"][0]) if len(current_row) > 0 else 0
      volatility = float(current_row["realized_volatility"][0]) if len(current_row) > 0 else 0.0
      
      # Current date
      current_date = current_row["date"][0] if len(current_row) > 0 else None
      
      window_dict = {
        "date": current_date,
        "price_sequence": price_sequence,
        "news": current_news,
        "direction": direction,
        "volatility": volatility,
      }
      
      windows.append(window_dict)
    
    return windows

  def process_data(self, price_df: pl.DataFrame, news_df: pl.DataFrame) -> list[dict[str, Any]]:
    """
    Complete preprocessing pipeline.
    
    Args:
      price_df: Raw price DataFrame
      news_df: Raw news DataFrame
      
    Returns:
      List of processed windows ready for model training
    """
    print("Computing returns and volatility...")
    price_df = self.compute_returns_and_volatility(price_df)
    
    print("Normalizing price features...")
    price_df = self.normalize_price_features(price_df)
    
    print("Aligning data by date...")
    aligned_df = self.align_data_by_date(price_df, news_df)
    
    print("Creating sliding windows...")
    windows = self.create_sliding_windows(aligned_df)
    
    print(f"Created {len(windows)} windows")
    
    return windows

  def get_feature_stats(self, windows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Get statistics about the processed features.
    
    Args:
      windows: List of processed windows
      
    Returns:
      Dictionary with feature statistics
    """
    if not windows:
      return {}
    
    # Collect all price sequences
    all_sequences = [w["price_sequence"] for w in windows]
    sequences_array = np.array(all_sequences)  # shape: (num_windows, window_size, num_features)
    
    # Collect volatilities and directions
    volatilities = [w["volatility"] for w in windows]
    directions = [w["direction"] for w in windows]
    
    stats = {
      "num_windows": len(windows),
      "window_size": len(windows[0]["price_sequence"]),
      "num_price_features": len(windows[0]["price_sequence"][0]),
      "price_sequence_shape": sequences_array.shape,
      "price_mean": np.mean(sequences_array),
      "price_std": np.std(sequences_array),
      "volatility_mean": np.mean(volatilities),
      "volatility_std": np.std(volatilities),
      "volatility_min": np.min(volatilities),
      "volatility_max": np.max(volatilities),
      "direction_distribution": {
        "up": sum(directions),
        "down": len(directions) - sum(directions),
        "up_ratio": sum(directions) / len(directions) if directions else 0,
      },
    }
    
    return stats


def main() -> None:
  """Test the preprocessor."""
  from .data_loader import CoinDeskDataLoader
  
  # Load data
  loader = CoinDeskDataLoader()
  price_df, news_df = loader.fetch_all_data()
  
  # Preprocess
  preprocessor = DataPreprocessor()
  windows = preprocessor.process_data(price_df, news_df)
  
  # Get stats
  stats = preprocessor.get_feature_stats(windows)
  
  print("Preprocessing Statistics:")
  for key, value in stats.items():
    print(f"  {key}: {value}")
  
  if windows:
    print("\nSample window:")
    sample = windows[0]
    print(f"  Date: {sample['date']}")
    print(f"  Price sequence shape: {np.array(sample['price_sequence']).shape}")
    print(f"  Direction: {sample['direction']}")
    print(f"  Volatility: {sample['volatility']:.4f}")
    print(f"  News headline: {sample['news']['headline'][:100]}...")


if __name__ == "__main__":
  main()