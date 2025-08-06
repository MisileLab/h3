"""Configuration file for ETH volatility and direction prediction model."""

import os
from pathlib import Path
from typing import Final

# Random seed for reproducibility
RANDOM_SEED: Final[int] = 42

# Project paths
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
EMBEDDINGS_DIR: Final[Path] = PROJECT_ROOT / "embeddings"
EXPERIMENTS_DIR: Final[Path] = PROJECT_ROOT / "experiments"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, EXPERIMENTS_DIR, NOTEBOOKS_DIR]:
  directory.mkdir(parents=True, exist_ok=True)

# API Configuration
COINDESK_BASE_URL: Final[str] = "https://data-api.coindesk.com"
COINDESK_API_KEY: Final[str | None] = os.getenv("COINDESK_API_KEY")  # Optional API key for enhanced rate limits
OPENAI_API_KEY: Final[str | None] = os.getenv("OPENAI_API_KEY")

# Data parameters
SYMBOL: Final[str] = "ETH"
PRICE_LOOKBACK_DAYS: Final[int] = 365  # Last 1 year of price data
NEWS_LOOKBACK_DAYS: Final[int] = 365  # Last 1 year of news data
WINDOW_SIZE: Final[int] = 10  # 10-day sliding window
NEWS_BATCH_SIZE: Final[int] = 100  # News articles per API call
PRICE_BATCH_DAYS: Final[int] = 5000  # Days per price API call (API limit)
EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
EMBEDDING_DIM: Final[int] = 1536  # text-embedding-3-small dimension

# Model hyperparameters
BATCH_SIZE: Final[int] = 32
LEARNING_RATE: Final[float] = 1e-3
NUM_EPOCHS: Final[int] = 100
EARLY_STOPPING_PATIENCE: Final[int] = 10

# Loss function weights
ALPHA_VOLATILITY_WEIGHT: Final[float] = 0.5  # Weight for volatility loss in combined loss

# Model architecture
LSTM_HIDDEN_SIZE: Final[int] = 64
LSTM_NUM_LAYERS: Final[int] = 2
LSTM_DROPOUT: Final[float] = 0.2

# News MLP configuration
NEWS_MLP_HIDDEN_SIZE: Final[int] = 128
NEWS_MLP_DROPOUT: Final[float] = 0.3

# Fusion layer configuration
FUSION_HIDDEN_SIZE: Final[int] = 256
FUSION_DROPOUT: Final[float] = 0.3

# Data split ratios
TRAIN_RATIO: Final[float] = 0.7
VAL_RATIO: Final[float] = 0.15
TEST_RATIO: Final[float] = 0.15

# File names
PRICE_DATA_FILE: Final[str] = "eth_ohlcv.parquet"
NEWS_DATA_FILE: Final[str] = "eth_news.parquet"
PRICE_DATA_JSON: Final[str] = "eth_ohlcv.json"  # Backup format
NEWS_DATA_JSON: Final[str] = "eth_news.json"   # Backup format
PROCESSED_DATA_FILE: Final[str] = "processed_data.safetensors"
MODEL_CHECKPOINT_FILE: Final[str] = "best_model.safetensors"
EMBEDDINGS_CACHE_FILE: Final[str] = "news_embeddings.safetensors"

# Device configuration
DEVICE: Final[str] = "cuda" if __name__ == "__main__" else "cpu"  # Will be set properly in training script

# Logging configuration
LOG_LEVEL: Final[str] = "INFO"
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Validation
if TRAIN_RATIO + VAL_RATIO + TEST_RATIO != 1.0:
  raise ValueError("Train, validation, and test ratios must sum to 1.0")

if OPENAI_API_KEY is None:
  import warnings
  warnings.warn("OPENAI_API_KEY environment variable not set. Embedding functionality will be limited.")