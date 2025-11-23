"""Temporal Q-Network: Computes distributional Q-values from latent sequences

Takes sequences of latent vectors and outputs distributional Q-values
for each action (stay/extract).
"""

import torch
import torch.nn as nn
import math
from typing import Literal


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input sequences

        Returns:
            [B, T, D] sequences with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class BiGRUTemporalEncoder(nn.Module):
    """Bidirectional GRU with attention pooling for temporal encoding"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim // 2,  # Bidirectional doubles the output
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] latent sequences

        Returns:
            [B, H] state embeddings
        """
        # GRU encoding
        gru_out, _ = self.gru(x)  # [B, T, H]
        gru_out = self.layer_norm(gru_out)

        # Attention pooling
        attention_weights = self.attention(gru_out)  # [B, T, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum
        context = torch.sum(gru_out * attention_weights, dim=1)  # [B, H]

        return context


class TransformerTemporalEncoder(nn.Module):
    """Transformer encoder for temporal encoding (alternative to GRU)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] latent sequences

        Returns:
            [B, H] state embeddings
        """
        # Project input
        x = self.input_proj(x)  # [B, T, H]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        encoded = self.transformer(x)  # [B, T, H]

        # Pool over time dimension
        pooled = self.pool(encoded.transpose(1, 2)).squeeze(-1)  # [B, H]

        return pooled


class TemporalQNet(nn.Module):
    """
    Temporal Q-Network that computes distributional Q-values.

    Takes sequences of latent vectors and outputs distributional Q-values
    for each action using quantile regression.

    Architecture:
    1. Temporal encoder (GRU or Transformer)
    2. State embedding
    3. Distributional Q-head for each action
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 512,
        num_actions: int = 2,  # stay, extract
        num_quantiles: int = 16,
        encoder_type: Literal["gru", "transformer"] = "gru",
        dropout: float = 0.1,
    ):
        """
        Args:
            latent_dim: Dimension of input latent vectors
            hidden_dim: Hidden dimension for temporal encoding
            num_actions: Number of actions (2: stay/extract)
            num_quantiles: Number of quantiles for distributional Q
            encoder_type: Type of temporal encoder
            dropout: Dropout probability
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.encoder_type = encoder_type

        # Temporal encoder
        if encoder_type == "gru":
            self.temporal_encoder = BiGRUTemporalEncoder(
                input_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
            )
        elif encoder_type == "transformer":
            self.temporal_encoder = TransformerTemporalEncoder(
                input_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_heads=8,
                num_layers=4,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Distributional Q-head
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions * num_quantiles),
        )

        # Quantile fractions (learnable or fixed)
        # Fixed uniform quantiles: τ_i = (i + 0.5) / K
        quantile_fractions = torch.linspace(0.0, 1.0, num_quantiles + 1)[:-1] + (
            0.5 / num_quantiles
        )
        self.register_buffer("quantile_fractions", quantile_fractions)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_seq: [B, T, D] latent sequences

        Returns:
            [B, num_actions, num_quantiles] distributional Q-values
        """
        # Encode temporal sequence
        state_emb = self.temporal_encoder(z_seq)  # [B, H]

        # Compute distributional Q-values
        q_flat = self.q_head(state_emb)  # [B, num_actions * num_quantiles]

        # Reshape to [B, num_actions, num_quantiles]
        q_dist = q_flat.view(-1, self.num_actions, self.num_quantiles)

        return q_dist

    def get_q_values(self, z_seq: torch.Tensor, quantile: float = 0.5) -> torch.Tensor:
        """
        Get Q-values at a specific quantile.

        Args:
            z_seq: [B, T, D] latent sequences
            quantile: Quantile to extract (0.5 = median)

        Returns:
            [B, num_actions] Q-values
        """
        q_dist = self.forward(z_seq)  # [B, A, K]

        # Find closest quantile index
        quantile_idx = torch.argmin(torch.abs(self.quantile_fractions - quantile))

        # Extract Q-values at that quantile
        q_values = q_dist[:, :, quantile_idx]  # [B, A]

        return q_values

    def get_ev(
        self, z_seq: torch.Tensor, quantile: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get Expected Values for stay and extract actions.

        Args:
            z_seq: [B, T, D] latent sequences
            quantile: Quantile to use (0.5 = median, <0.5 = conservative, >0.5 = optimistic)

        Returns:
            (ev_stay, ev_extract): [B] tensors of EV for each action
        """
        q_values = self.get_q_values(z_seq, quantile)  # [B, 2]
        ev_stay = q_values[:, 0]
        ev_extract = q_values[:, 1]
        return ev_stay, ev_extract


def test_qnet():
    """Test temporal Q-net with dummy data"""
    print("Testing TemporalQNet...")

    # Test GRU version
    qnet_gru = TemporalQNet(
        latent_dim=512,
        hidden_dim=512,
        num_actions=2,
        num_quantiles=16,
        encoder_type="gru",
    )
    print(f"QNet (GRU) created: hidden_dim={qnet_gru.hidden_dim}, quantiles={qnet_gru.num_quantiles}")

    z_seq = torch.randn(4, 32, 512)  # [B=4, T=32, D=512]
    q_dist = qnet_gru(z_seq)
    print(f"GRU output: input {z_seq.shape} -> q_dist {q_dist.shape}")
    assert q_dist.shape == (4, 2, 16)

    # Test get_q_values
    q_values = qnet_gru.get_q_values(z_seq, quantile=0.5)
    print(f"Q-values at median: {q_values.shape}")
    assert q_values.shape == (4, 2)

    # Test get_ev
    ev_stay, ev_extract = qnet_gru.get_ev(z_seq, quantile=0.5)
    print(f"EV: stay {ev_stay.shape}, extract {ev_extract.shape}")
    assert ev_stay.shape == (4,) and ev_extract.shape == (4,)

    # Test Transformer version
    qnet_transformer = TemporalQNet(
        latent_dim=512,
        hidden_dim=512,
        encoder_type="transformer",
    )
    q_dist_tf = qnet_transformer(z_seq)
    print(f"Transformer output: {q_dist_tf.shape}")
    assert q_dist_tf.shape == (4, 2, 16)

    print("✓ TemporalQNet tests passed")


if __name__ == "__main__":
    test_qnet()
