"""Training loop for EV model

Implements distributional Q-learning with quantile regression loss.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from arcx.ml.model import EVModel
from arcx.ml.utils import save_checkpoint, load_checkpoint, save_model_safetensors
from arcx.device import device_manager
from arcx.config import config

logger = logging.getLogger(__name__)


class QuantileHuberLoss(nn.Module):
    """
    Quantile Huber loss for distributional Q-learning.

    Combines quantile regression with Huber loss for robustness.
    """

    def __init__(self, quantiles: torch.Tensor, kappa: float = 1.0):
        """
        Args:
            quantiles: [K] tensor of quantile fractions
            kappa: Huber loss threshold
        """
        super().__init__()
        self.quantiles = quantiles
        self.kappa = kappa

    def forward(
        self,
        q_dist: torch.Tensor,
        target: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute quantile Huber loss.

        Args:
            q_dist: [B, A, K] predicted distributional Q-values
            target: [B, 1] target returns
            action: [B, 1] action indices

        Returns:
            Scalar loss
        """
        B, A, K = q_dist.shape

        # Select Q-values for taken actions: [B, K]
        action_expanded = action.unsqueeze(-1).expand(B, 1, K)  # [B, 1, K]
        q_selected = q_dist.gather(1, action_expanded).squeeze(1)  # [B, K]

        # Expand target: [B, 1] -> [B, K]
        target_expanded = target.expand(B, K)  # [B, K]

        # TD error
        td_error = target_expanded - q_selected  # [B, K]

        # Huber loss
        huber_loss = torch.where(
            td_error.abs() <= self.kappa,
            0.5 * td_error.pow(2),
            self.kappa * (td_error.abs() - 0.5 * self.kappa),
        )

        # Quantile weights
        quantiles = self.quantiles.to(q_dist.device).view(1, K)
        quantile_weight = torch.abs(quantiles - (td_error < 0).float())

        # Quantile Huber loss
        loss = quantile_weight * huber_loss
        loss = loss.mean()

        return loss


class Trainer:
    """
    Trainer for EV model.

    Handles training loop, validation, checkpointing.
    """

    def __init__(
        self,
        model: EVModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Args:
            model: EVModel to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            learning_rate: Learning rate
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device_manager.device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        # Loss function
        quantiles = self.model.qnet.quantile_fractions
        self.criterion = QuantileHuberLoss(quantiles=quantiles, kappa=1.0)

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir or (config.model.encoder_path.parent / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        logger.info(f"Trainer initialized: lr={learning_rate}, device={self.device}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for z_seq, action, target in pbar:
            # Move to device
            z_seq = z_seq.to(self.device)
            action = action.to(self.device)
            target = target.to(self.device)

            # Forward pass
            q_dist = self.model.forward_latents(z_seq)  # [B, A, K]

            # Compute loss
            loss = self.criterion(q_dist, target, action)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Stats
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}

    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for z_seq, action, target in tqdm(self.val_loader, desc="Validation"):
                z_seq = z_seq.to(self.device)
                action = action.to(self.device)
                target = target.to(self.device)

                # Forward
                q_dist = self.model.forward_latents(z_seq)

                # Loss
                loss = self.criterion(q_dist, target, action)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}

    def train(self, num_epochs: int, save_every: int = 5):
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log
            metrics = {**train_metrics, **val_metrics}
            logger.info(f"Epoch {epoch}: {metrics}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint()

            # Save best model
            if val_metrics and val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_best_model()
                logger.info(f"New best model: val_loss={self.best_val_loss:.4f}")

        logger.info("Training complete")

        # Save final model
        self.save_final_model()

    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"epoch_{self.current_epoch}"
        save_checkpoint(
            self.model,
            self.optimizer,
            self.current_epoch,
            checkpoint_path,
            additional_state={"best_val_loss": self.best_val_loss},
        )
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint"""
        state = load_checkpoint(
            self.model,
            self.optimizer,
            checkpoint_path,
            device=self.device,
        )
        self.current_epoch = state.get("epoch", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        logger.info(f"Checkpoint loaded: epoch={self.current_epoch}")

    def save_best_model(self):
        """Save best model weights"""
        save_model_safetensors(
            self.model.encoder,
            config.model.encoder_path,
            metadata={"epoch": str(self.current_epoch), "type": "best"},
        )
        save_model_safetensors(
            self.model.qnet,
            config.model.qnet_path,
            metadata={"epoch": str(self.current_epoch), "type": "best"},
        )
        logger.info("Best model saved")

    def save_final_model(self):
        """Save final model weights"""
        encoder_path = config.model.encoder_path.parent / "encoder_final.safetensors"
        qnet_path = config.model.qnet_path.parent / "qnet_final.safetensors"

        save_model_safetensors(
            self.model.encoder,
            encoder_path,
            metadata={"epoch": str(self.current_epoch), "type": "final"},
        )
        save_model_safetensors(
            self.model.qnet,
            qnet_path,
            metadata={"epoch": str(self.current_epoch), "type": "final"},
        )
        logger.info("Final model saved")


def test_trainer():
    """Test trainer with dummy data"""
    import tempfile
    from arcx.data.logger import DataLogger
    from arcx.data.dataset import create_dataloaders

    print("Testing Trainer...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy data
        logger_inst = DataLogger(log_dir=Path(tmpdir))

        for run_idx in range(5):
            logger_inst.start_run(f"run_{run_idx}", "test")

            for decision_idx in range(20):
                z_seq = torch.randn(32, 512)
                logger_inst.log_decision(
                    decision_idx=decision_idx,
                    t_sec=decision_idx * 10.0,
                    action="stay" if decision_idx < 19 else "extract",
                    z_seq=z_seq,
                )

            logger_inst.end_run(
                final_loot_value=1000.0 + run_idx * 200,
                total_time_sec=200.0,
                success=True,
            )

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            data_dir=Path(tmpdir),
            batch_size=16,
            train_split=0.8,
            num_workers=0,
        )

        # Create model
        model = EVModel(
            encoder_backbone="resnet34",
            latent_dim=512,
            encoder_pretrained=False,
            hidden_dim=256,
            num_quantiles=8,
            temporal_encoder="gru",
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=1e-3,
            checkpoint_dir=Path(tmpdir) / "checkpoints",
        )

        # Train for a few epochs
        trainer.train(num_epochs=2, save_every=1)

        print("✓ Trainer test passed")


if __name__ == "__main__":
    test_trainer()
