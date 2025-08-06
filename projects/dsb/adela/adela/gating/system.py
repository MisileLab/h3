"""Dynamic gating system for expert selection."""

from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adela.experts.base import ExpertBase


class GatingNetwork(nn.Module):
    """Network for dynamically selecting and weighting experts."""

    def __init__(
        self, 
        num_experts: int,
        hidden_size: int = 256
    ) -> None:
        """Initialize the gating network.

        Args:
            num_experts: Number of experts to gate.
            hidden_size: Size of hidden layers.
        """
        super().__init__()
        
        # Input features: board tensor (flattened) + additional features
        input_size = 12 * 8 * 8 + 8
        
        # Gating network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_experts)
        
    def forward(
        self, 
        board_tensor: torch.Tensor, 
        additional_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the gating network.

        Args:
            board_tensor: Tensor of shape (batch_size, 12, 8, 8) representing the board state.
            additional_features: Tensor of shape (batch_size, 8) with additional features.

        Returns:
            Tensor of shape (batch_size, num_experts) with expert weights.
        """
        batch_size = board_tensor.shape[0]
        
        # Flatten board tensor
        board_flat = board_tensor.view(batch_size, -1)
        
        # Concatenate board and additional features
        x = torch.cat([board_flat, additional_features], dim=1)
        
        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Apply softmax to get expert weights
        weights = F.softmax(x, dim=1)
        
        return weights


class MixtureOfExperts(nn.Module):
    """Mixture of experts model combining multiple expert networks."""

    def __init__(
        self, 
        experts: Dict[str, ExpertBase],
        gating_hidden_size: int = 256
    ) -> None:
        """Initialize the mixture of experts model.

        Args:
            experts: Dictionary of expert networks.
            gating_hidden_size: Size of hidden layers in the gating network.
        """
        super().__init__()
        
        self.experts = nn.ModuleDict(experts)
        self.expert_names = list(experts.keys())
        self.num_experts = len(experts)
        
        # Create gating network
        self.gating_network = GatingNetwork(self.num_experts, gating_hidden_size)
        
    def forward(
        self, 
        board_tensor: torch.Tensor, 
        additional_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the mixture of experts model.

        Args:
            board_tensor: Tensor of shape (batch_size, 12, 8, 8) representing the board state.
            additional_features: Tensor of shape (batch_size, 8) with additional features.

        Returns:
            Tuple of (policy_logits, value, expert_weights):
                - policy_logits: Tensor of shape (batch_size, num_moves) with move probabilities.
                - value: Tensor of shape (batch_size, 1) with position evaluation.
                - expert_weights: Tensor of shape (batch_size, num_experts) with expert weights.
        """
        batch_size = board_tensor.shape[0]
        
        # Get expert weights from gating network
        expert_weights = self.gating_network(board_tensor, additional_features)
        
        # Initialize policy and value tensors
        policy_logits = torch.zeros(batch_size, 1968, device=board_tensor.device)  # Maximum possible moves
        value = torch.zeros(batch_size, 1, device=board_tensor.device)
        
        # Run each expert and combine outputs
        for i, expert_name in enumerate(self.expert_names):
            expert = self.experts[expert_name]
            expert_policy, expert_value = expert(board_tensor, additional_features)
            
            # Weight the expert outputs by the gating weights
            # Reshape weights for broadcasting
            weights = expert_weights[:, i].view(batch_size, 1)
            
            policy_logits += weights * expert_policy
            value += weights * expert_value
            
        return policy_logits, value, expert_weights
    
    def get_expert_contributions(
        self, 
        board_tensor: torch.Tensor, 
        additional_features: torch.Tensor
    ) -> Dict[str, float]:
        """Get the contribution of each expert for a given position.

        Args:
            board_tensor: Tensor of shape (1, 12, 8, 8) representing the board state.
            additional_features: Tensor of shape (1, 8) with additional features.

        Returns:
            Dictionary mapping expert names to their weights.
        """
        # Get expert weights
        with torch.no_grad():
            expert_weights = self.gating_network(board_tensor, additional_features)
        
        # Convert to dictionary
        contributions = {
            name: weight.item()
            for name, weight in zip(self.expert_names, expert_weights[0])
        }
        
        return contributions
