# ETH Volatility and Direction Prediction

A complete end-to-end machine learning pipeline for predicting Ethereum (ETH) daily volatility and price direction using historical OHLCV data and news sentiment analysis.

## Project Overview

This project implements a dual-task neural network that predicts:

1. **Direction (Binary Classification)**: Whether ETH's daily close price will go up (1) or down (0) compared to its opening price
2. **Volatility (Regression)**: The realized volatility percentage over the same daily period

### Key Features

- 🔄 **Data Pipeline**: Automated fetching of ETH OHLCV data (1 year) and news (1 year) from CoinDesk API with sliding window pagination for both data sources
- 🧠 **Advanced Model**: LSTM price encoder + MLP news encoder with fusion architecture
- 📊 **News Analysis**: OpenAI embeddings for news sentiment with intelligent caching
- ⚡ **Efficient Storage**: Parquet for dataframes, SafeTensors for model checkpointing, embeddings cache, and processed data
- 📈 **Comprehensive Evaluation**: Detailed metrics and visualization for both tasks
- 🎯 **Time-based Splits**: Proper temporal data splitting to avoid look-ahead bias

## Architecture

```
Input Features (10-day window):
├── Price Sequence: OHLCV data → LSTM Encoder
└── News Embedding: Daily news → MLP Encoder
                                    ↓
                            Fusion Module
                                    ↓
                    ┌─────────────────────────────┐
                    ↓                             ↓
            Direction Head                Volatility Head
            (Sigmoid)                     (Linear)
                    ↓                             ↓
            Classification                Regression
            (BCE Loss)                    (MSE Loss)
```

## Installation

### Prerequisites

- Python 3.13.5
- UV package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd spectator
   ```

2. **Install dependencies**:
   ```bash
   uv install
   ```

3. **Set up environment variables**:
   ```bash
   # Create .env file or set environment variables
   export OPENAI_API_KEY="your-openai-api-key-here"
   export COINDESK_API_KEY="your-coindesk-api-key-here"  # Optional - for enhanced rate limits
   ```

4. **Activate the virtual environment**:
   ```bash
   uv shell
   ```

## Usage

### Quick Start

#### 1. Data Collection and Preprocessing
```bash
# Test data loading
uv run python -m src.data_loader

# Test preprocessing
uv run python -m src.preprocessing
```

#### 2. Generate News Embeddings
```bash
# Generate and cache news embeddings
uv run python -m src.embedder
```

#### 3. Train the Model
```bash
# Train the complete model
uv run python -m src.train
```

#### 4. Evaluate Results
```bash
# Evaluate trained model and generate plots
uv run python -m src.evaluate
```

### Configuration

All hyperparameters and settings are defined in `src/config.py`:

```python
# Data parameters
PRICE_LOOKBACK_DAYS = 365   # Days of historical price data (1 year)
NEWS_LOOKBACK_DAYS = 365    # Days of historical news data (1 year)  
WINDOW_SIZE = 10            # Sliding window size for model input
PRICE_BATCH_DAYS = 30       # Days per price API batch (API limit)
NEWS_BATCH_SIZE = 100       # Articles per news API batch
EMBEDDING_MODEL = "text-embedding-3-small"

# Model architecture
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
NEWS_MLP_HIDDEN_SIZE = 128
FUSION_HIDDEN_SIZE = 256

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
ALPHA_VOLATILITY_WEIGHT = 0.5  # Balance between tasks
```

### Advanced Usage

#### Custom Data Sources

Replace the data loader with your own:

```python
from src.data_loader import CoinDeskDataLoader

class CustomDataLoader(CoinDeskDataLoader):
    def fetch_price_data(self):
        # Your custom implementation
        pass
```

#### Model Architecture Modifications

Extend the model in `src/model.py`:

```python
from src.model import ETHVolatilityModel

class CustomModel(ETHVolatilityModel):
    def __init__(self, price_input_size):
        super().__init__(price_input_size)
        # Add your custom layers
```

## Project Structure

```
spectator/
├── data/                   # Data storage
│   ├── raw/               # Raw OHLCV and news data
│   │   ├── *.parquet     # Efficient dataframe storage (primary)
│   │   ├── *.json        # Human-readable backup format
│   │   └── *.safetensors # Processed numerical data
│   └── processed/         # Preprocessed data
├── embeddings/            # Cached news embeddings (safetensors)
├── experiments/           # Model checkpoints and results (safetensors)
├── notebooks/             # Jupyter notebooks for EDA
├── src/                   # Source code
│   ├── config.py         # Configuration and hyperparameters
│   ├── data_loader.py    # Data fetching from APIs
│   ├── preprocessing.py  # Data preprocessing and alignment
│   ├── embedder.py       # News embedding generation
│   ├── dataset.py        # PyTorch Dataset implementation
│   ├── model.py          # Neural network architecture
│   ├── train.py          # Training pipeline
│   └── evaluate.py       # Evaluation and metrics
├── pyproject.toml        # Project dependencies
└── README.md            # This file
```

### Data Storage Formats

The pipeline uses optimized storage formats for different data types:

- **📊 Parquet Files**: Primary storage for dataframes (price/news data)
  - Fast loading with Polars
  - Compressed and efficient
  - Schema preservation
- **🗃️ JSON Files**: Backup format for human readability  
  - Easy debugging and inspection
  - Cross-platform compatibility
- **⚡ SafeTensors**: Numerical data and model storage
  - Fast loading for ML workflows
  - Memory-mapped access
  - Safe cross-platform format
  - Used for: embeddings, processed windows, model checkpoints

## Model Details

### Input Features

**Price Sequence (10 days × 5 features)**:
- Open price percentage change
- High price percentage change  
- Low price percentage change
- Close price percentage change
- Volume percentage change

**News Embedding (1536 dimensions)**:
- OpenAI text-embedding-3-small model
- Combined headline and content text
- Cached using SafeTensors for efficiency

### Architecture Components

1. **Price Encoder**: 2-layer LSTM with dropout
2. **News Encoder**: 2-layer MLP with ReLU activation
3. **Fusion Module**: Concatenation + 2-layer MLP
4. **Task Heads**: 
   - Direction: MLP + Sigmoid (binary classification)
   - Volatility: MLP + Linear (regression)

### Loss Function

Combined loss with configurable weighting:
```
Loss = (1 - α) × BCE(direction) + α × MSE(volatility)
```

Where α = 0.5 by default (equal weighting).

## Evaluation Metrics

### Direction Prediction
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Class-wise performance metrics  
- **AUC-ROC**: Area under the ROC curve
- **Confusion Matrix**: Classification breakdown

### Volatility Prediction
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **Correlation**: Pearson correlation coefficient

## Results and Visualization

After training, the evaluation script generates:

1. **Confusion Matrix**: Direction prediction accuracy breakdown
2. **Probability Distributions**: Direction prediction confidence analysis
3. **Scatter Plots**: Actual vs predicted volatility
4. **Residual Analysis**: Prediction error distribution
5. **Time Series**: Error patterns over time

All plots are saved to `experiments/plots/`.

## API Requirements

### CoinDesk Data API
- **Price Data Endpoint**: `https://data-api.coindesk.com/spot/v1/historical/days`
- **News Search Endpoint**: `https://data-api.coindesk.com/news/v1/search`
- **Sliding Window**: Price data fetched in 30-day batches, news data using `to_ts` pagination
- **Data Range**: Both price and news data collected for 1 year (365 days)
- **Rate Limits**: Enhanced limits with API key, standard limits without
- **Authentication**: Optional API key for enhanced access (recommended)
- **Documentation**: [CoinDesk Data API](https://developers.coindesk.com/documentation/data-api/introduction)

### OpenAI API
- **Model**: text-embedding-3-small
- **Rate Limits**: 3,000 RPM (Tier 1)
- **Cost**: ~$0.02 per 1M tokens
- **Authentication**: API key required

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code  
uv run ruff check

# Type checking
uv run mypy src/
```

### Adding New Features

1. **New Data Sources**: Extend `DataLoader` class
2. **Model Components**: Add to `model.py` 
3. **Evaluation Metrics**: Extend `ModelEvaluator`
4. **Preprocessing Steps**: Add to `DataPreprocessor`

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```python
   # Reduce batch size in config.py
   BATCH_SIZE = 16  # or smaller
   ```

2. **OpenAI API Rate Limits**:
   ```python
   # Embeddings are cached - subsequent runs use cache
   # Check embeddings/ directory for cached files
   ```

3. **CoinDesk API Issues**:
   ```python
   # If API structure changes, fallback data will be used
   # Check console output for "Warning: Using simulated data"
   # Set COINDESK_API_KEY for enhanced rate limits
   ```

4. **Data Loading Errors**:
   ```python
   # Use cached data if API fails
   loader.fetch_all_data(use_cache=True)
   ```

### Performance Optimization

1. **Faster Training**:
   - Use GPU if available
   - Increase batch size (if memory allows)
   - Enable mixed precision training

2. **Memory Efficiency**:
   - Reduce window size
   - Use gradient checkpointing
   - Process data in smaller chunks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{eth_volatility_predictor,
  title={ETH Volatility and Direction Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/spectator}
}
```

## Acknowledgments

- CoinDesk for providing cryptocurrency data
- OpenAI for embedding models
- PyTorch and Polars communities for excellent libraries

---

For questions and support, please open an issue on GitHub.
