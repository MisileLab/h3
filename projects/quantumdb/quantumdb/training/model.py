"""
Learnable Product Quantization model implementation.

This module implements the LearnablePQ model which combines neural network
based encoding with product quantization for efficient vector compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""

    def __init__(self, layers: list, activation: str = "relu", dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # No activation on final layer
                if activation == "relu":
                    self.layers.append(nn.ReLU())
                elif activation == "gelu":
                    self.layers.append(nn.GELU())
                elif activation == "tanh":
                    self.layers.append(nn.Tanh())
                if dropout > 0:
                    self.layers.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class LearnablePQ(nn.Module):
    """
    Learnable Product Quantization model.

    This model learns to compress high-dimensional vectors into compact representations
    using product quantization with learnable codebooks.
    """

    def __init__(
        self,
        input_dim: int = 768,
        target_dim: int = 256,
        n_subvectors: int = 16,
        codebook_size: int = 256,
        encoder_layers: Optional[list] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.target_dim = target_dim
        self.n_subvectors = n_subvectors
        self.codebook_size = codebook_size
        self.subvector_dim = target_dim // n_subvectors

        # Validate dimensions
        assert target_dim % n_subvectors == 0, (
            "target_dim must be divisible by n_subvectors"
        )

        # Default encoder architecture
        if encoder_layers is None:
            encoder_layers = [input_dim, 512, 384, target_dim]

        # Encoder network
        self.encoder = MLP(encoder_layers, activation="gelu", dropout=dropout)

        # Learnable codebooks
        self.codebooks = nn.Parameter(
            torch.randn(n_subvectors, codebook_size, self.subvector_dim) * 0.1
        )

        # Temperature for soft assignment
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input vectors to compressed representation."""
        return self.encoder(x)

    def quantize(
        self, encoded: torch.Tensor, hard: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize encoded vectors using product quantization.

        Args:
            encoded: Encoded vectors [batch_size, target_dim]
            hard: Whether to use hard assignment (True) or soft assignment (False)

        Returns:
            quantized: Quantized vectors [batch_size, target_dim]
            codes: Code assignments [batch_size, n_subvectors]
        """
        batch_size = encoded.size(0)

        # Reshape for subvector processing
        encoded_subvectors = encoded.view(
            batch_size, self.n_subvectors, self.subvector_dim
        )  # [batch_size, n_subvectors, subvector_dim]

        # Compute distances to codebook entries
        # Expand for broadcasting
        encoded_expanded = encoded_subvectors.unsqueeze(
            2
        )  # [batch_size, n_subvectors, 1, subvector_dim]
        codebooks_expanded = self.codebooks.unsqueeze(
            0
        )  # [1, n_subvectors, codebook_size, subvector_dim]

        # Compute squared distances
        distances = torch.sum(
            (encoded_expanded - codebooks_expanded) ** 2, dim=-1
        )  # [batch_size, n_subvectors, codebook_size]

        if hard:
            # Hard assignment - take argmin
            codes = torch.argmin(distances, dim=-1)  # [batch_size, n_subvectors]

            # Gather quantized vectors
            quantized_subvectors = torch.stack(
                [self.codebooks[i, codes[:, i]] for i in range(self.n_subvectors)],
                dim=1,
            )  # [batch_size, n_subvectors, subvector_dim]

        else:
            # Soft assignment using softmax
            soft_assignments = F.softmax(-distances / self.temperature, dim=-1)

            # Weighted combination of codebook entries
            quantized_subvectors = torch.sum(
                soft_assignments.unsqueeze(-1) * codebooks_expanded, dim=2
            )  # [batch_size, n_subvectors, subvector_dim]

            # For soft assignment, return expected codes (not used in backprop)
            codes = torch.argmax(soft_assignments, dim=-1)

        # Reshape back to original format
        quantized = quantized_subvectors.view(batch_size, self.target_dim)

        return quantized, codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode from code assignments to vectors.

        Args:
            codes: Code assignments [batch_size, n_subvectors]

        Returns:
            decoded: Decoded vectors [batch_size, target_dim]
        """
        batch_size = codes.size(0)

        # Gather vectors from codebooks
        decoded_subvectors = torch.stack(
            [self.codebooks[i, codes[:, i]] for i in range(self.n_subvectors)], dim=1
        )  # [batch_size, n_subvectors, subvector_dim]

        # Reshape to original format
        decoded = decoded_subvectors.view(batch_size, self.target_dim)

        return decoded

    def forward(self, x: torch.Tensor, return_codes: bool = False) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input vectors [batch_size, input_dim]
            return_codes: Whether to return code assignments

        Returns:
            output: Quantized vectors [batch_size, target_dim]
            codes: (optional) Code assignments [batch_size, n_subvectors]
        """
        encoded = self.encode(x)
        quantized, codes = self.quantize(encoded)

        if return_codes:
            return quantized, codes
        return quantized

    def get_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        original_bits = self.input_dim * 32  # Assuming float32
        compressed_bits = self.n_subvectors * 8  # 8 bits per subvector (256 codes)
        return original_bits / compressed_bits

    def get_model_size(self) -> int:
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
