"""Base expert network implementation."""

from abc import ABC, abstractmethod
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertBase(nn.Module, ABC):
  """Base class for all expert networks."""

  def __init__(self, name: str) -> None:
    """Initialize the expert.

    Args:
      name: Expert name.
    """
    super().__init__() # pyright: ignore[reportUnknownMemberType]
    self.name: str = name
  
  @abstractmethod
  @override
  def forward(
    self,
    board_tensor: torch.Tensor,
    additional_features: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass of the expert network.

    Args:
      board_tensor: Tensor of shape (batch_size, 12, 8, 8) representing the board state.
      additional_features: Tensor of shape (batch_size, 8) with additional features.

    Returns:
      Tuple of (policy_logits, value):
        - policy_logits: Tensor of shape (batch_size, num_moves) with move probabilities.
        - value: Tensor of shape (batch_size, 1) with position evaluation.
    """
    pass
  
  def get_metadata(self) -> dict[str, str]:
    """Get expert metadata.

    Returns:
      Dictionary with expert metadata.
    """
    return {
      "name": self.name,
      "type": self.__class__.__name__,
    }

class ConvolutionalExpert(ExpertBase):
  """Convolutional neural network expert."""

  def __init__(
    self,
    name: str,
    num_filters: int = 256,
    num_blocks: int = 10,
    policy_head_hidden: int = 256,
    value_head_hidden: int = 256
  ) -> None:
    """Initialize the convolutional expert.

    Args:
      name: Expert name.
      num_filters: Number of filters in convolutional layers.
      num_blocks: Number of residual blocks.
      policy_head_hidden: Number of hidden units in policy head.
      value_head_hidden: Number of hidden units in value head.
    """
    super().__init__(name)
    # Initial convolution
    self.conv_initial = nn.Conv2d(12, num_filters, kernel_size=3, padding=1)
    self.bn_initial = nn.BatchNorm2d(num_filters)
    # Residual blocks
    self.residual_blocks = nn.ModuleList([
      ResidualBlock(num_filters) for _ in range(num_blocks)
    ])
    # Additional features processing
    self.additional_features_fc = nn.Linear(8, 32)
    # Policy head
    self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
    self.policy_bn = nn.BatchNorm2d(32)
    self.policy_fc1 = nn.Linear(32 * 8 * 8, policy_head_hidden)
    # Output size is flexible and will be set during forward pass based on legal moves
    # Value head
    self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
    self.value_bn = nn.BatchNorm2d(32)
    self.value_fc1 = nn.Linear(32 * 8 * 8 + 32, value_head_hidden)  # +32 for additional features
    self.value_fc2 = nn.Linear(value_head_hidden, 1)
  
  def forward(
    self,
    board_tensor: torch.Tensor,
    additional_features: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass of the expert network.

    Args:
      board_tensor: Tensor of shape (batch_size, 12, 8, 8) representing the board state.
      additional_features: Tensor of shape (batch_size, 8) with additional features.

    Returns:
      Tuple of (policy_logits, value):
        - policy_logits: Tensor of shape (batch_size, num_moves) with move probabilities.
        - value: Tensor of shape (batch_size, 1) with position evaluation.
    """
    batch_size = board_tensor.shape[0]
    # Initial convolution
    x = F.relu(self.bn_initial(self.conv_initial(board_tensor)))
    # Residual blocks
    for block in self.residual_blocks:
      x = block(x)
    # Process additional features
    additional_features = F.relu(self.additional_features_fc(additional_features))
    # Policy head
    policy = F.relu(self.policy_bn(self.policy_conv(x)))
    policy = policy.view(batch_size, -1)
    policy = F.relu(self.policy_fc1(policy))
    # For now, we'll output a fixed size policy vector
    # In practice, this would be dynamically sized based on legal moves
    policy_logits = torch.zeros(batch_size, 1968, device=board_tensor.device)  # Maximum possible moves
    # Value head
    value = F.relu(self.value_bn(self.value_conv(x)))
    value = value.view(batch_size, -1)
    value = torch.cat([value, additional_features], dim=1)
    value = F.relu(self.value_fc1(value))
    value = torch.tanh(self.value_fc2(value))  # Tanh to get value in [-1, 1]
    return policy_logits, value

class ResidualBlock(nn.Module):
  """Residual block for the convolutional expert."""

  def __init__(self, num_filters: int) -> None:
    """Initialize the residual block.

    Args:
      num_filters: Number of filters in convolutional layers.
    """
    super().__init__()
    self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(num_filters)
    self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(num_filters)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the residual block.

    Args:
      x: Input tensor.
    Returns:
      Output tensor.
    """
    residual = x
    x = F.relu(self.bn1(self.conv1(x)))
    x = self.bn2(self.conv2(x))
    x += residual
    x = F.relu(x)
    return x
