"""
Tests for training modules.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

# Import modules to test
from quantumdb.training.model import LearnablePQ
from quantumdb.training.losses import QuantizationLoss, ReconstructionLoss
from quantumdb.training.trainer import Trainer


class TestLearnablePQ:
    """Test cases for LearnablePQ model."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = LearnablePQ(
            input_dim=768,
            target_dim=256,
            n_subvectors=16,
            codebook_size=256,
        )

    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.input_dim == 768
        assert self.model.target_dim == 256
        assert self.model.n_subvectors == 16
        assert self.model.codebook_size == 256
        assert self.model.subvector_dim == 16

        # Check codebook shape
        assert self.model.codebooks.shape == (16, 256, 16)

    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, 768)

        # Forward pass
        output = self.model(x)

        # Check output shape
        assert output.shape == (batch_size, 256)

        # Check output is finite
        assert torch.isfinite(output).all()

    def test_forward_with_codes(self):
        """Test forward pass with code return."""
        batch_size = 4
        x = torch.randn(batch_size, 768)

        # Forward pass with codes
        output, codes = self.model(x, return_codes=True)

        # Check shapes
        assert output.shape == (batch_size, 256)
        assert codes.shape == (batch_size, 16)

        # Check codes are valid integers
        assert (codes >= 0).all()
        assert (codes < 256).all()

    def test_encode_decode(self):
        """Test encoding and decoding."""
        batch_size = 4
        x = torch.randn(batch_size, 768)

        # Encode
        encoded = self.model.encode(x)
        assert encoded.shape == (batch_size, 256)

        # Quantize
        quantized, codes = self.model.quantize(encoded)
        assert quantized.shape == (batch_size, 256)
        assert codes.shape == (batch_size, 16)

        # Decode
        decoded = self.model.decode(codes)
        assert decoded.shape == (batch_size, 256)

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        ratio = self.model.get_compression_ratio()
        expected_ratio = (768 * 32) / (16 * 8)  # float32 vs int8 per subvector
        assert abs(ratio - expected_ratio) < 0.01

    def test_model_size(self):
        """Test model size calculation."""
        size = self.model.get_model_size()
        assert size > 0
        assert isinstance(size, int)


class TestLosses:
    """Test cases for loss functions."""

    def test_quantization_loss(self):
        """Test quantization loss."""
        loss_fn = QuantizationLoss()

        batch_size = 4
        encoded = torch.randn(batch_size, 256, requires_grad=True)
        quantized = torch.randn(batch_size, 256)

        loss = loss_fn(encoded, quantized)

        # Check loss is scalar and requires grad
        assert loss.dim() == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)

    def test_reconstruction_loss(self):
        """Test reconstruction loss."""
        loss_fn = ReconstructionLoss(loss_type="mse")

        batch_size = 4
        original = torch.randn(batch_size, 768)
        reconstructed = torch.randn(batch_size, 768)

        loss = loss_fn(original, reconstructed)

        # Check loss is scalar
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_reconstruction_loss_cosine(self):
        """Test reconstruction loss with cosine similarity."""
        loss_fn = ReconstructionLoss(loss_type="cosine")

        batch_size = 4
        original = torch.randn(batch_size, 768)
        reconstructed = torch.randn(batch_size, 768)
        target = torch.ones(batch_size)

        loss = loss_fn(original, reconstructed)

        # Check loss is scalar
        assert loss.dim() == 0
        assert torch.isfinite(loss)


class TestTrainer:
    """Test cases for Trainer class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = LearnablePQ(
            input_dim=768,
            target_dim=256,
            n_subvectors=16,
            codebook_size=256,
        )

        # Generate synthetic data
        np.random.seed(42)
        self.train_data = np.random.randn(1000, 768).astype(np.float32)
        self.val_data = np.random.randn(200, 768).astype(np.float32)

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = Trainer(
            self.model,
            experiment_name="test_experiment",
            use_wandb=False,
        )

        assert trainer.model == self.model
        assert trainer.experiment_name == "test_experiment"
        assert not trainer.use_wandb

    def test_create_data_loaders(self):
        """Test data loader creation."""
        trainer = Trainer(self.model, use_wandb=False)

        train_loader, val_loader = trainer.create_data_loaders(
            self.train_data,
            self.val_data,
            batch_size=32,
        )

        # Check train loader
        assert len(train_loader) > 0
        batch = next(iter(train_loader))
        assert batch.shape[0] <= 32
        assert batch.shape[1] == 768

        # Check val loader
        assert val_loader is not None
        val_batch = next(iter(val_loader))
        assert val_batch.shape[1] == 768

    def test_train_epoch(self):
        """Test training epoch."""
        trainer = Trainer(self.model, use_wandb=False)

        train_loader, _ = trainer.create_data_loaders(self.train_data, batch_size=32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        metrics = trainer.train_epoch(train_loader, optimizer, loss_fn)

        # Check metrics
        assert "total_loss" in metrics
        assert isinstance(metrics["total_loss"], float)
        assert metrics["total_loss"] > 0

    def test_fit(self):
        """Test full training fit."""
        trainer = Trainer(
            self.model,
            experiment_name="test_fit",
            use_wandb=False,
        )

        history = trainer.fit(
            train_data=self.train_data,
            val_data=self.val_data,
            epochs=2,
            batch_size=32,
            learning_rate=1e-3,
            save_best=False,
        )

        # Check history
        assert "train_loss" in history
        assert len(history["train_loss"]) == 2
        assert all(loss > 0 for loss in history["train_loss"])

    def test_save_load_model(self):
        """Test model saving and loading."""
        trainer = Trainer(self.model, use_wandb=False)

        # Save model
        models_dir = Path("test_models")
        models_dir.mkdir(exist_ok=True)

        trainer.save_model(str(models_dir), "test_model")

        # Check files exist
        model_file = models_dir / "test_model.safetensors"
        metadata_file = models_dir / "test_model_metadata.json"

        assert model_file.exists()
        assert metadata_file.exists()

        # Load model
        new_model = LearnablePQ(
            input_dim=768,
            target_dim=256,
            n_subvectors=16,
            codebook_size=256,
        )

        trainer.load_model(str(model_file))

        # Check models have same parameters
        for p1, p2 in zip(self.model.parameters(), trainer.model.parameters()):
            assert torch.allclose(p1, p2)

        # Cleanup
        model_file.unlink()
        metadata_file.unlink()
        models_dir.rmdir()


if __name__ == "__main__":
    pytest.main([__file__])
