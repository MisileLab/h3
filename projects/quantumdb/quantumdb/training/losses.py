"""
Loss functions for training learnable product quantization models.

This module implements various loss functions suitable for training
vector compression models including reconstruction, quantization,
and metric learning losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class QuantizationLoss(nn.Module):
    """
    Quantization loss that encourages encoded vectors to be close to codebook entries.

    This loss minimizes the distance between encoded vectors and their quantized
    representations, encouraging better codebook utilization.
    """

    def __init__(self, commitment_cost: float = 0.25):
        super().__init__()
        self.commitment_cost = commitment_cost

    def forward(self, encoded: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        """
        Calculate quantization loss.

        Args:
            encoded: Original encoded vectors [batch_size, target_dim]
            quantized: Quantized vectors [batch_size, target_dim]

        Returns:
            loss: Quantization loss
        """
        e_latent_loss = F.mse_loss(quantized.detach(), encoded)
        q_latent_loss = F.mse_loss(quantized, encoded.detach())

        return q_latent_loss + self.commitment_cost * e_latent_loss


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for training autoencoder-style models.

    This loss ensures that the compressed representation preserves
    the essential information from the original input.
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "cosine":
            self.criterion = nn.CosineEmbeddingLoss()
        elif loss_type == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate reconstruction loss.

        Args:
            original: Original input vectors [batch_size, input_dim]
            reconstructed: Reconstructed vectors [batch_size, input_dim]

        Returns:
            loss: Reconstruction loss
        """
        if self.loss_type == "cosine":
            # For cosine loss, we need target labels (1 for similar)
            target = torch.ones(original.size(0), device=original.device)
            return self.criterion(original, reconstructed, target)
        else:
            return self.criterion(original, reconstructed)


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.

    This loss encourages similar vectors to be closer together and
    dissimilar vectors to be farther apart in the compressed space.
    """

    def __init__(self, margin: float = 1.0, p: float = 2.0):
        super().__init__()
        self.margin = margin
        self.p = p

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate triplet loss.

        Args:
            anchor: Anchor vectors [batch_size, target_dim]
            positive: Positive vectors [batch_size, target_dim]
            negative: Negative vectors [batch_size, target_dim]

        Returns:
            loss: Triplet loss
        """
        distance_positive = F.pairwise_distance(anchor, positive, p=self.p)
        distance_negative = F.pairwise_distance(anchor, negative, p=self.p)

        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for metric learning.

    This loss learns embeddings where similar samples are close and
    dissimilar samples are separated by at least a margin.
    """

    def __init__(self, margin: float = 2.0, p: float = 2.0):
        super().__init__()
        self.margin = margin
        self.p = p

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate contrastive loss.

        Args:
            x1: First set of vectors [batch_size, target_dim]
            x2: Second set of vectors [batch_size, target_dim]
            label: Similarity labels (1 for similar, 0 for dissimilar) [batch_size]

        Returns:
            loss: Contrastive loss
        """
        distance = F.pairwise_distance(x1, x2, p=self.p)

        # Similar pairs: minimize distance
        # Dissimilar pairs: maximize distance up to margin
        loss_similar = label * distance**2
        loss_dissimilar = (1 - label) * F.relu(self.margin - distance) ** 2

        return 0.5 * (loss_similar + loss_dissimilar).mean()


class DiversityLoss(nn.Module):
    """
    Diversity loss to encourage uniform codebook usage.

    This loss encourages the model to use all codebook entries equally,
    preventing codebook collapse.
    """

    def __init__(self, n_subvectors: int, codebook_size: int):
        super().__init__()
        self.n_subvectors = n_subvectors
        self.codebook_size = codebook_size
        self.target_entropy = -torch.log(torch.tensor(1.0 / codebook_size))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Calculate diversity loss.

        Args:
            codes: Code assignments [batch_size, n_subvectors]

        Returns:
            loss: Diversity loss
        """
        batch_size = codes.size(0)
        diversity_loss = 0.0

        for i in range(self.n_subvectors):
            # Count usage of each code in this subvector
            code_counts = torch.bincount(
                codes[:, i], minlength=self.codebook_size
            ).float()

            # Convert to probabilities
            probs = code_counts / batch_size

            # Add small epsilon to avoid log(0)
            eps = 1e-8
            probs = probs + eps
            probs = probs / probs.sum()

            # Calculate entropy
            entropy = -torch.sum(probs * torch.log(probs))

            # Loss is difference from target entropy (maximum entropy)
            diversity_loss += (self.target_entropy - entropy) ** 2

        return diversity_loss / self.n_subvectors


class CombinedLoss(nn.Module):
    """
    Combined loss that weights multiple loss functions.

    This allows flexible combination of different loss components
    for optimal training performance.
    """

    def __init__(
        self,
        quantization_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        triplet_weight: float = 0.0,
        diversity_weight: float = 0.1,
        margin: float = 1.0,
        n_subvectors: int = 16,
        codebook_size: int = 256,
    ):
        super().__init__()

        self.quantization_weight = quantization_weight
        self.reconstruction_weight = reconstruction_weight
        self.triplet_weight = triplet_weight
        self.diversity_weight = diversity_weight

        self.quantization_loss = QuantizationLoss()
        self.reconstruction_loss = ReconstructionLoss()
        self.triplet_loss = TripletLoss(margin=margin)
        self.diversity_loss = DiversityLoss(n_subvectors, codebook_size)

    def forward(
        self,
        original: Optional[torch.Tensor] = None,
        encoded: Optional[torch.Tensor] = None,
        quantized: Optional[torch.Tensor] = None,
        reconstructed: Optional[torch.Tensor] = None,
        anchor: Optional[torch.Tensor] = None,
        positive: Optional[torch.Tensor] = None,
        negative: Optional[torch.Tensor] = None,
        codes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate combined loss.

        Args:
            original: Original input vectors
            encoded: Encoded vectors
            quantized: Quantized vectors
            reconstructed: Reconstructed vectors
            anchor: Anchor vectors for triplet loss
            positive: Positive vectors for triplet loss
            negative: Negative vectors for triplet loss
            codes: Code assignments for diversity loss

        Returns:
            total_loss: Combined loss
            loss_components: Dictionary of individual loss components
        """
        loss_components = {}
        total_loss = 0.0

        # Quantization loss
        if (
            encoded is not None
            and quantized is not None
            and self.quantization_weight > 0
        ):
            q_loss = self.quantization_loss(encoded, quantized)
            loss_components["quantization"] = q_loss
            total_loss += self.quantization_weight * q_loss

        # Reconstruction loss
        if (
            original is not None
            and reconstructed is not None
            and self.reconstruction_weight > 0
        ):
            r_loss = self.reconstruction_loss(original, reconstructed)
            loss_components["reconstruction"] = r_loss
            total_loss += self.reconstruction_weight * r_loss

        # Triplet loss
        if (
            anchor is not None
            and positive is not None
            and negative is not None
            and self.triplet_weight > 0
        ):
            t_loss = self.triplet_loss(anchor, positive, negative)
            loss_components["triplet"] = t_loss
            total_loss += self.triplet_weight * t_loss

        # Diversity loss
        if codes is not None and self.diversity_weight > 0:
            d_loss = self.diversity_loss(codes)
            loss_components["diversity"] = d_loss
            total_loss += self.diversity_weight * d_loss

        return total_loss, loss_components
