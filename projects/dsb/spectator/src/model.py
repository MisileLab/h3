"""Neural network model for ETH volatility and direction prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from .config import (
  RANDOM_SEED,
  EMBEDDING_DIM,
  LSTM_HIDDEN_SIZE,
  LSTM_NUM_LAYERS,
  LSTM_DROPOUT,
  NEWS_MLP_HIDDEN_SIZE,
  NEWS_MLP_DROPOUT,
  FUSION_HIDDEN_SIZE,
  FUSION_DROPOUT,
)


class PriceEncoder(nn.Module):
  """LSTM encoder for price sequence data."""

  def __init__(self, input_size: int, hidden_size: int = LSTM_HIDDEN_SIZE, num_layers: int = LSTM_NUM_LAYERS, dropout: float = LSTM_DROPOUT) -> None:
    """
    Initialize the price encoder.
    
    Args:
      input_size: Number of price features per timestep
      hidden_size: LSTM hidden size
      num_layers: Number of LSTM layers
      dropout: Dropout rate
    """
    super().__init__()
    
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    
    self.lstm = nn.LSTM(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      dropout=dropout if num_layers > 1 else 0,
      batch_first=True,
      bidirectional=False,
    )
    
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the price encoder.
    
    Args:
      x: Price sequence tensor [batch_size, seq_len, features]
      
    Returns:
      Encoded price features [batch_size, hidden_size]
    """
    # LSTM forward pass
    # x shape: [batch_size, seq_len, features]
    lstm_out, (hidden, _) = self.lstm(x)
    
    # Use the last hidden state
    # hidden shape: [num_layers, batch_size, hidden_size]
    last_hidden = hidden[-1]  # [batch_size, hidden_size]
    
    # Apply dropout
    encoded = self.dropout(last_hidden)
    
    return encoded


class NewsEncoder(nn.Module):
  """MLP encoder for news embedding data."""

  def __init__(self, input_size: int = EMBEDDING_DIM, hidden_size: int = NEWS_MLP_HIDDEN_SIZE, dropout: float = NEWS_MLP_DROPOUT) -> None:
    """
    Initialize the news encoder.
    
    Args:
      input_size: Size of news embedding
      hidden_size: Hidden layer size
      dropout: Dropout rate
    """
    super().__init__()
    
    self.mlp = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_size, hidden_size // 2),
      nn.ReLU(),
      nn.Dropout(dropout),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the news encoder.
    
    Args:
      x: News embedding tensor [batch_size, embedding_dim]
      
    Returns:
      Encoded news features [batch_size, hidden_size // 2]
    """
    return self.mlp(x)


class FusionModule(nn.Module):
  """Fusion module to combine price and news features."""

  def __init__(self, price_features: int, news_features: int, hidden_size: int = FUSION_HIDDEN_SIZE, dropout: float = FUSION_DROPOUT) -> None:
    """
    Initialize the fusion module.
    
    Args:
      price_features: Number of price features
      news_features: Number of news features
      hidden_size: Hidden layer size
      dropout: Dropout rate
    """
    super().__init__()
    
    total_input_size = price_features + news_features
    
    self.fusion_layers = nn.Sequential(
      nn.Linear(total_input_size, hidden_size),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_size, hidden_size // 2),
      nn.ReLU(),
      nn.Dropout(dropout),
    )

  def forward(self, price_features: torch.Tensor, news_features: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the fusion module.
    
    Args:
      price_features: Encoded price features [batch_size, price_features]
      news_features: Encoded news features [batch_size, news_features]
      
    Returns:
      Fused features [batch_size, hidden_size // 2]
    """
    # Concatenate features
    combined = torch.cat([price_features, news_features], dim=1)
    
    # Pass through fusion layers
    fused = self.fusion_layers(combined)
    
    return fused


class DirectionHead(nn.Module):
  """Classification head for direction prediction."""

  def __init__(self, input_size: int) -> None:
    """
    Initialize the direction head.
    
    Args:
      input_size: Number of input features
    """
    super().__init__()
    
    self.classifier = nn.Sequential(
      nn.Linear(input_size, input_size // 2),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(input_size // 2, 1),
      nn.Sigmoid(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the direction head.
    
    Args:
      x: Fused features [batch_size, features]
      
    Returns:
      Direction probabilities [batch_size, 1]
    """
    return self.classifier(x)


class VolatilityHead(nn.Module):
  """Regression head for volatility prediction."""

  def __init__(self, input_size: int) -> None:
    """
    Initialize the volatility head.
    
    Args:
      input_size: Number of input features
    """
    super().__init__()
    
    self.regressor = nn.Sequential(
      nn.Linear(input_size, input_size // 2),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(input_size // 2, 1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the volatility head.
    
    Args:
      x: Fused features [batch_size, features]
      
    Returns:
      Volatility predictions [batch_size, 1]
    """
    return self.regressor(x)


class ETHVolatilityModel(nn.Module):
  """Complete model for ETH volatility and direction prediction."""

  def __init__(self, price_input_size: int) -> None:
    """
    Initialize the complete model.
    
    Args:
      price_input_size: Number of price features per timestep
    """
    super().__init__()
    
    # Encoders
    self.price_encoder = PriceEncoder(price_input_size)
    self.news_encoder = NewsEncoder()
    
    # Calculate fusion input sizes
    price_output_size = LSTM_HIDDEN_SIZE
    news_output_size = NEWS_MLP_HIDDEN_SIZE // 2
    
    # Fusion module
    self.fusion = FusionModule(price_output_size, news_output_size)
    
    # Prediction heads
    fusion_output_size = FUSION_HIDDEN_SIZE // 2
    self.direction_head = DirectionHead(fusion_output_size)
    self.volatility_head = VolatilityHead(fusion_output_size)
    
    # Initialize weights
    self._initialize_weights()

  def _initialize_weights(self) -> None:
    """Initialize model weights with a fixed random seed."""
    torch.manual_seed(RANDOM_SEED)
    
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
          if 'weight' in name:
            nn.init.xavier_uniform_(param)
          elif 'bias' in name:
            nn.init.zeros_(param)

  def forward(self, price_seq: torch.Tensor, news_emb: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Forward pass through the complete model.
    
    Args:
      price_seq: Price sequence tensor [batch_size, seq_len, features]
      news_emb: News embedding tensor [batch_size, embedding_dim]
      
    Returns:
      Dictionary with:
        - direction: Direction probabilities [batch_size, 1]
        - volatility: Volatility predictions [batch_size, 1]
    """
    # Encode inputs
    price_features = self.price_encoder(price_seq)
    news_features = self.news_encoder(news_emb)
    
    # Fuse features
    fused_features = self.fusion(price_features, news_features)
    
    # Make predictions
    direction_pred = self.direction_head(fused_features)
    volatility_pred = self.volatility_head(fused_features)
    
    return {
      "direction": direction_pred,
      "volatility": volatility_pred,
    }

  def get_model_info(self) -> dict[str, Any]:
    """Get information about the model architecture."""
    total_params = sum(p.numel() for p in self.parameters())
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    return {
      "total_parameters": total_params,
      "trainable_parameters": trainable_params,
      "price_encoder_params": sum(p.numel() for p in self.price_encoder.parameters()),
      "news_encoder_params": sum(p.numel() for p in self.news_encoder.parameters()),
      "fusion_params": sum(p.numel() for p in self.fusion.parameters()),
      "direction_head_params": sum(p.numel() for p in self.direction_head.parameters()),
      "volatility_head_params": sum(p.numel() for p in self.volatility_head.parameters()),
    }


class CombinedLoss(nn.Module):
  """Combined loss function for direction and volatility prediction."""

  def __init__(self, alpha: float = 0.5) -> None:
    """
    Initialize the combined loss.
    
    Args:
      alpha: Weight for volatility loss (direction loss weight = 1 - alpha)
    """
    super().__init__()
    self.alpha = alpha
    self.bce_loss = nn.BCELoss()
    self.mse_loss = nn.MSELoss()

  def forward(self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Compute combined loss.
    
    Args:
      predictions: Dictionary with direction and volatility predictions
      targets: Dictionary with direction and volatility targets
      
    Returns:
      Dictionary with individual and combined losses
    """
    # Individual losses
    direction_loss = self.bce_loss(predictions["direction"], targets["direction"])
    volatility_loss = self.mse_loss(predictions["volatility"], targets["volatility"])
    
    # Combined loss
    combined_loss = (1 - self.alpha) * direction_loss + self.alpha * volatility_loss
    
    return {
      "direction_loss": direction_loss,
      "volatility_loss": volatility_loss,
      "combined_loss": combined_loss,
    }


def create_model(price_input_size: int) -> ETHVolatilityModel:
  """
  Create and initialize the ETH volatility model.
  
  Args:
    price_input_size: Number of price features per timestep
    
  Returns:
    Initialized model
  """
  model = ETHVolatilityModel(price_input_size)
  return model


def main() -> None:
  """Test the model implementation."""
  # Set random seed
  torch.manual_seed(RANDOM_SEED)
  
  # Model parameters
  batch_size = 4
  seq_len = 10
  price_features = 5
  
  # Create model
  model = create_model(price_features)
  
  # Print model info
  model_info = model.get_model_info()
  print("Model Information:")
  for key, value in model_info.items():
    print(f"  {key}: {value:,}")
  
  # Create dummy input data
  price_seq = torch.randn(batch_size, seq_len, price_features)
  news_emb = torch.randn(batch_size, EMBEDDING_DIM)
  
  print(f"\nInput shapes:")
  print(f"  Price sequence: {price_seq.shape}")
  print(f"  News embedding: {news_emb.shape}")
  
  # Forward pass
  model.eval()
  with torch.no_grad():
    predictions = model(price_seq, news_emb)
  
  print(f"\nOutput shapes:")
  for key, tensor in predictions.items():
    print(f"  {key}: {tensor.shape}")
  
  # Test loss function
  targets = {
    "direction": torch.randint(0, 2, (batch_size, 1)).float(),
    "volatility": torch.rand(batch_size, 1),
  }
  
  loss_fn = CombinedLoss(alpha=0.5)
  losses = loss_fn(predictions, targets)
  
  print(f"\nLoss values:")
  for key, loss in losses.items():
    print(f"  {key}: {loss.item():.4f}")
  
  # Test model in training mode
  model.train()
  predictions_train = model(price_seq, news_emb)
  print(f"\nTraining mode output shapes:")
  for key, tensor in predictions_train.items():
    print(f"  {key}: {tensor.shape}")


if __name__ == "__main__":
  main()