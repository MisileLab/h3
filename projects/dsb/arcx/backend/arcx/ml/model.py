"""Integrated EV Model: Frame Encoder + Temporal Q-Net

Combines frame encoder and Q-network for end-to-end inference.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional

from arcx.ml.encoder import FrameEncoder
from arcx.ml.qnet import TemporalQNet


class EVModel(nn.Module):
    """
    Complete EV prediction model.

    Combines:
    1. Frame Encoder: converts frames to latent vectors
    2. Temporal Q-Net: computes distributional Q-values from latent sequences

    Can operate in two modes:
    - End-to-end: raw frames → latent → Q-values
    - Latent-only: pre-computed latents → Q-values (for faster inference)
    """

    def __init__(
        self,
        # Encoder params
        encoder_backbone: Literal["resnet34", "resnet50"] = "resnet34",
        latent_dim: int = 512,
        encoder_pretrained: bool = True,
        freeze_encoder: bool = False,
        # Q-net params
        hidden_dim: int = 512,
        num_actions: int = 2,
        num_quantiles: int = 16,
        temporal_encoder: Literal["gru", "transformer"] = "gru",
        dropout: float = 0.1,
    ):
        """
        Args:
            encoder_backbone: ResNet variant for frame encoder
            latent_dim: Latent vector dimension
            encoder_pretrained: Use ImageNet pretrained encoder
            freeze_encoder: Freeze encoder during training
            hidden_dim: Hidden dimension for Q-net
            num_actions: Number of actions (2: stay/extract)
            num_quantiles: Number of quantiles for distributional Q
            temporal_encoder: Temporal encoder type
            dropout: Dropout probability
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        # Frame encoder
        self.encoder = FrameEncoder(
            backbone=encoder_backbone,
            latent_dim=latent_dim,
            pretrained=encoder_pretrained,
            freeze_backbone=freeze_encoder,
        )

        # Temporal Q-network
        self.qnet = TemporalQNet(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_quantiles=num_quantiles,
            encoder_type=temporal_encoder,
            dropout=dropout,
        )

    def forward_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass from raw frames.

        Args:
            frames: [B, T, 3, H, W] frame sequences

        Returns:
            [B, num_actions, num_quantiles] distributional Q-values
        """
        # Encode frames to latent sequences
        z_seq = self.encoder.encode_sequence(frames)  # [B, T, D]

        # Compute Q-values
        q_dist = self.qnet(z_seq)  # [B, A, K]

        return q_dist

    def forward_latents(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from pre-computed latent sequences.
        Faster for real-time inference when latents are cached.

        Args:
            z_seq: [B, T, D] latent sequences

        Returns:
            [B, num_actions, num_quantiles] distributional Q-values
        """
        return self.qnet(z_seq)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode frames to latent vectors.

        Args:
            frames: [B, T, 3, H, W] or [B, 3, H, W]

        Returns:
            [B, T, D] or [B, D] latent vectors
        """
        if frames.ndim == 5:
            return self.encoder.encode_sequence(frames)
        elif frames.ndim == 4:
            return self.encoder.encode_batch(frames)
        else:
            raise ValueError(f"Invalid frames shape: {frames.shape}")

    def predict_ev(
        self,
        frames: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        quantile: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict Expected Values for stay and extract actions.

        Args:
            frames: [B, T, 3, H, W] raw frames (optional if latents provided)
            latents: [B, T, D] pre-computed latents (optional if frames provided)
            quantile: Quantile to use (0.5=median, <0.5=conservative, >0.5=optimistic)

        Returns:
            (ev_stay, ev_extract, delta_ev): [B] tensors
        """
        if latents is None and frames is None:
            raise ValueError("Either frames or latents must be provided")

        # Get latent sequences
        if latents is None:
            z_seq = self.encoder.encode_sequence(frames)
        else:
            z_seq = latents

        # Get EV values
        ev_stay, ev_extract = self.qnet.get_ev(z_seq, quantile)

        # Compute delta
        delta_ev = ev_stay - ev_extract

        return ev_stay, ev_extract, delta_ev

    def get_recommendation(
        self,
        frames: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        quantile: float = 0.5,
        threshold: float = 20.0,
    ) -> tuple[str, float, float, float]:
        """
        Get action recommendation with human-readable output.

        Args:
            frames: [B, T, 3, H, W] raw frames
            latents: [B, T, D] latents
            quantile: Risk profile quantile
            threshold: Delta threshold for strong recommendations

        Returns:
            (action, ev_stay, ev_extract, delta_ev)
            action: "stay", "extract", or "neutral"
        """
        ev_stay, ev_extract, delta_ev = self.predict_ev(frames, latents, quantile)

        # Take first batch element (assumes batch_size=1 for real-time inference)
        ev_stay = ev_stay.item()
        ev_extract = ev_extract.item()
        delta_ev = delta_ev.item()

        # Determine recommendation
        if delta_ev > threshold:
            action = "stay"
        elif delta_ev < -threshold:
            action = "extract"
        else:
            action = "neutral"

        return action, ev_stay, ev_extract, delta_ev

    @property
    def device(self) -> torch.device:
        """Get device of model parameters"""
        return next(self.parameters()).device


def test_evmodel():
    """Test complete EV model"""
    print("Testing EVModel...")

    model = EVModel(
        encoder_backbone="resnet34",
        latent_dim=512,
        encoder_pretrained=False,  # Faster for testing
        hidden_dim=512,
        num_quantiles=16,
        temporal_encoder="gru",
    )
    print(f"Model created: latent_dim={model.latent_dim}, quantiles={model.num_quantiles}")

    # Test end-to-end forward
    frames = torch.randn(2, 32, 3, 225, 400)  # [B=2, T=32, C=3, H=225, W=400]
    q_dist = model.forward_frames(frames)
    print(f"End-to-end: frames {frames.shape} -> q_dist {q_dist.shape}")
    assert q_dist.shape == (2, 2, 16)

    # Test latent-only forward
    latents = torch.randn(2, 32, 512)
    q_dist_latent = model.forward_latents(latents)
    print(f"Latent-only: latents {latents.shape} -> q_dist {q_dist_latent.shape}")
    assert q_dist_latent.shape == (2, 2, 16)

    # Test EV prediction
    ev_stay, ev_extract, delta_ev = model.predict_ev(frames=frames, quantile=0.5)
    print(f"EV prediction: stay {ev_stay}, extract {ev_extract}, delta {delta_ev}")
    assert ev_stay.shape == (2,)

    # Test recommendation
    frames_single = frames[:1]  # Single batch
    action, ev_s, ev_e, delta = model.get_recommendation(frames=frames_single, quantile=0.5)
    print(f"Recommendation: {action}, EV_stay={ev_s:.2f}, EV_extract={ev_e:.2f}, delta={delta:.2f}")
    assert action in ["stay", "extract", "neutral"]

    print("✓ EVModel tests passed")


if __name__ == "__main__":
    test_evmodel()
