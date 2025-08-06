"""News embedder using OpenAI API with safetensors caching."""

import hashlib
import warnings
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from openai import OpenAI
from safetensors.numpy import load_file, save_file
from tqdm import tqdm

from .config import (
  OPENAI_API_KEY,
  EMBEDDINGS_DIR,
  EMBEDDING_MODEL,
  EMBEDDING_DIM,
  EMBEDDINGS_CACHE_FILE,
)


class NewsEmbedder:
  """Handles news embedding generation and caching."""

  def __init__(self) -> None:
    """Initialize the embedder."""
    if OPENAI_API_KEY is None:
      raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    self.client = OpenAI(api_key=OPENAI_API_KEY)
    self.cache_file = EMBEDDINGS_DIR / EMBEDDINGS_CACHE_FILE
    self._cache: dict[str, np.ndarray] = {}
    self._load_cache()

  def _generate_text_key(self, text: str) -> str:
    """Generate a unique key for caching based on text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

  def _load_cache(self) -> None:
    """Load cached embeddings from safetensors file."""
    if self.cache_file.exists():
      try:
        self._cache = load_file(str(self.cache_file))
        print(f"Loaded {len(self._cache)} cached embeddings")
      except Exception as e:
        warnings.warn(f"Failed to load embedding cache: {e}")
        self._cache = {}
    else:
      self._cache = {}

  def _save_cache(self) -> None:
    """Save embeddings cache to safetensors file."""
    try:
      save_file(self._cache, str(self.cache_file))
    except Exception as e:
      warnings.warn(f"Failed to save embedding cache: {e}")

  def _get_embedding_from_api(self, text: str) -> np.ndarray:
    """
    Get embedding from OpenAI API.
    
    Args:
      text: Text to embed
      
    Returns:
      Embedding vector as numpy array
    """
    try:
      response = self.client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        encoding_format="float"
      )
      
      embedding = np.array(response.data[0].embedding, dtype=np.float32)
      
      if embedding.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Expected embedding dimension {EMBEDDING_DIM}, got {embedding.shape[0]}")
      
      return embedding
      
    except Exception as e:
      raise RuntimeError(f"Failed to get embedding from OpenAI API: {e}")

  def get_embedding(self, text: str) -> np.ndarray:
    """
    Get embedding for text, using cache if available.
    
    Args:
      text: Text to embed
      
    Returns:
      Embedding vector as numpy array
    """
    if not text.strip():
      # Return zero vector for empty text
      return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    text_key = self._generate_text_key(text)
    
    # Check cache first
    if text_key in self._cache:
      return self._cache[text_key]
    
    # Get from API and cache
    embedding = self._get_embedding_from_api(text)
    self._cache[text_key] = embedding
    
    # Save cache periodically (every 10 new embeddings)
    if len(self._cache) % 10 == 0:
      self._save_cache()
    
    return embedding

  def embed_news_text(self, headline: str, content: str) -> np.ndarray:
    """
    Embed news text combining headline and content.
    
    Args:
      headline: News headline
      content: News content/body
      
    Returns:
      Combined embedding vector
    """
    # Combine headline and content with separator
    combined_text = f"{headline.strip()}\n\n{content.strip()}"
    
    # Truncate if too long (OpenAI has token limits)
    max_chars = 8000  # Conservative limit to stay under token limits
    if len(combined_text) > max_chars:
      combined_text = combined_text[:max_chars] + "..."
    
    return self.get_embedding(combined_text)

  def process_windows(self, windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Process windows to add news embeddings.
    
    Args:
      windows: List of windows from preprocessing
      
    Returns:
      Windows with added news_embedding field
    """
    print("Generating news embeddings...")
    
    processed_windows = []
    
    for window in tqdm(windows, desc="Embedding news"):
      news = window["news"]
      headline = news.get("headline", "")
      content = news.get("content", "")
      
      # Get embedding for this news
      news_embedding = self.embed_news_text(headline, content)
      
      # Add embedding to window
      processed_window = window.copy()
      processed_window["news_embedding"] = news_embedding
      
      processed_windows.append(processed_window)
    
    # Save final cache
    self._save_cache()
    
    return processed_windows

  def aggregate_daily_embeddings(self, embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Aggregate multiple embeddings for a single day using mean pooling.
    
    Args:
      embeddings: List of embedding vectors
      
    Returns:
      Aggregated embedding vector
    """
    if not embeddings:
      return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    # Stack embeddings and compute mean
    stacked = np.stack(embeddings, axis=0)
    return np.mean(stacked, axis=0).astype(np.float32)

  def embed_dataframe_news(self, df: pl.DataFrame) -> pl.DataFrame:
    """
    Add embedding column to a dataframe with news data.
    
    Args:
      df: DataFrame with headline and content columns
      
    Returns:
      DataFrame with added embedding column
    """
    embeddings = []
    
    print("Generating embeddings for dataframe...")
    
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Embedding"):
      headline = row.get("headline", "")
      content = row.get("content", "")
      
      embedding = self.embed_news_text(headline, content)
      embeddings.append(embedding.tolist())  # Convert to list for polars
    
    # Save cache
    self._save_cache()
    
    # Add embeddings to dataframe
    df_with_embeddings = df.with_columns([
      pl.Series("embedding", embeddings)
    ])
    
    return df_with_embeddings

  def get_cache_stats(self) -> dict[str, Any]:
    """Get statistics about the embedding cache."""
    return {
      "cache_size": len(self._cache),
      "cache_file_exists": self.cache_file.exists(),
      "cache_file_path": str(self.cache_file),
      "embedding_dimension": EMBEDDING_DIM,
      "embedding_model": EMBEDDING_MODEL,
    }


def main() -> None:
  """Test the embedder."""
  from .data_loader import CoinDeskDataLoader
  from .preprocessing import DataPreprocessor
  
  # Load and preprocess data
  loader = CoinDeskDataLoader()
  price_df, news_df = loader.fetch_all_data()
  
  preprocessor = DataPreprocessor()
  windows = preprocessor.process_data(price_df, news_df)
  
  # Initialize embedder
  try:
    embedder = NewsEmbedder()
    
    # Get cache stats
    stats = embedder.get_cache_stats()
    print("Embedder Statistics:")
    for key, value in stats.items():
      print(f"  {key}: {value}")
    
    # Process a few sample windows
    sample_size = min(3, len(windows))
    sample_windows = windows[:sample_size]
    
    print(f"\nProcessing {sample_size} sample windows...")
    processed_windows = embedder.process_windows(sample_windows)
    
    # Show results
    for i, window in enumerate(processed_windows):
      embedding = window["news_embedding"]
      print(f"\nWindow {i}:")
      print(f"  Date: {window['date']}")
      print(f"  News headline: {window['news']['headline'][:100]}...")
      print(f"  Embedding shape: {embedding.shape}")
      print(f"  Embedding mean: {np.mean(embedding):.4f}")
      print(f"  Embedding std: {np.std(embedding):.4f}")
    
  except ValueError as e:
    print(f"Error: {e}")
    print("Please set the OPENAI_API_KEY environment variable to test embeddings.")


if __name__ == "__main__":
  main()