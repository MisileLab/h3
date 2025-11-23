"""Frame Encoder: Converts game frames to latent vectors

Uses ResNet backbone (pretrained on ImageNet) to extract spatial features
from downscaled game frames.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Literal


class FrameEncoder(nn.Module):
    """
    Frame encoder that converts single frames to latent vectors.

    Architecture:
    - ResNet backbone (34 or 50) pretrained on ImageNet
    - Remove final FC layer
    - Add custom projection head to target dimension

    Input: [B, 3, H, W] - RGB frames
    Output: [B, D] - Latent vectors
    """

    def __init__(
        self,
        backbone: Literal["resnet34", "resnet50"] = "resnet34",
        latent_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            backbone: ResNet variant to use
            latent_dim: Output latent vector dimension
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone weights during training
        """
        super().__init__()

        self.backbone_name = backbone
        self.latent_dim = latent_dim

        # Load backbone
        if backbone == "resnet34":
            resnet = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove final FC layer and avgpool (we'll use adaptive pooling)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Keep up to layer4

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Adaptive pooling + projection head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] input frames

        Returns:
            [B, latent_dim] latent vectors
        """
        # Extract features
        features = self.backbone(x)  # [B, C, H', W']

        # Global pooling
        pooled = self.global_pool(features)  # [B, C, 1, 1]
        pooled = pooled.flatten(1)  # [B, C]

        # Project to latent space
        z = self.projection(pooled)  # [B, latent_dim]

        return z

    def encode_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for encoding a batch of frames.

        Args:
            frames: [B, 3, H, W] frames

        Returns:
            [B, D] latent vectors
        """
        return self.forward(frames)

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence of frames.

        Args:
            frames: [B, T, 3, H, W] frame sequences

        Returns:
            [B, T, D] latent sequences
        """
        B, T, C, H, W = frames.shape
        # Reshape to [B*T, C, H, W]
        frames_flat = frames.view(B * T, C, H, W)

        # Encode
        z_flat = self.forward(frames_flat)  # [B*T, D]

        # Reshape back to [B, T, D]
        z_seq = z_flat.view(B, T, self.latent_dim)

        return z_seq

    @property
    def output_dim(self) -> int:
        """Output dimension of latent vectors"""
        return self.latent_dim


def test_encoder():
    """Test frame encoder with dummy data"""
    print("Testing FrameEncoder...")

    encoder = FrameEncoder(backbone="resnet34", latent_dim=512, pretrained=False)
    print(f"Encoder created: {encoder.backbone_name}, output_dim={encoder.output_dim}")

    # Test single batch
    x = torch.randn(4, 3, 225, 400)  # [B=4, C=3, H=225, W=400]
    z = encoder(x)
    print(f"Single batch: input {x.shape} -> output {z.shape}")
    assert z.shape == (4, 512)

    # Test sequence encoding
    x_seq = torch.randn(2, 32, 3, 225, 400)  # [B=2, T=32, C=3, H=225, W=400]
    z_seq = encoder.encode_sequence(x_seq)
    print(f"Sequence: input {x_seq.shape} -> output {z_seq.shape}")
    assert z_seq.shape == (2, 32, 512)

    print("✓ FrameEncoder tests passed")


if __name__ == "__main__":
    test_encoder()
