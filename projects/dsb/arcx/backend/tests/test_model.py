"""Tests for ML models

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import tempfile
from pathlib import Path

from arcx.ml.encoder import FrameEncoder
from arcx.ml.qnet import TemporalQNet
from arcx.ml.model import EVModel
from arcx.ml.utils import save_model_safetensors, load_model_safetensors


class TestFrameEncoder:
    """Test suite for FrameEncoder"""

    def test_encoder_creation(self):
        """Test encoder initialization"""
        encoder = FrameEncoder(backbone="resnet34", latent_dim=512, pretrained=False)
        assert encoder.latent_dim == 512
        assert encoder.backbone_name == "resnet34"

    def test_encoder_forward(self):
        """Test encoder forward pass"""
        encoder = FrameEncoder(backbone="resnet34", latent_dim=512, pretrained=False)
        x = torch.randn(4, 3, 225, 400)
        z = encoder(x)
        assert z.shape == (4, 512)

    def test_encoder_sequence(self):
        """Test sequence encoding"""
        encoder = FrameEncoder(backbone="resnet34", latent_dim=512, pretrained=False)
        x_seq = torch.randn(2, 32, 3, 225, 400)
        z_seq = encoder.encode_sequence(x_seq)
        assert z_seq.shape == (2, 32, 512)

    def test_encoder_resnet50(self):
        """Test with ResNet-50 backbone"""
        encoder = FrameEncoder(backbone="resnet50", latent_dim=1024, pretrained=False)
        x = torch.randn(2, 3, 225, 400)
        z = encoder(x)
        assert z.shape == (2, 1024)


class TestTemporalQNet:
    """Test suite for TemporalQNet"""

    def test_qnet_gru(self):
        """Test GRU-based Q-net"""
        qnet = TemporalQNet(
            latent_dim=512,
            hidden_dim=512,
            num_actions=2,
            num_quantiles=16,
            encoder_type="gru",
        )
        z_seq = torch.randn(4, 32, 512)
        q_dist = qnet(z_seq)
        assert q_dist.shape == (4, 2, 16)

    def test_qnet_transformer(self):
        """Test Transformer-based Q-net"""
        qnet = TemporalQNet(
            latent_dim=512,
            hidden_dim=512,
            encoder_type="transformer",
        )
        z_seq = torch.randn(4, 32, 512)
        q_dist = qnet(z_seq)
        assert q_dist.shape == (4, 2, 16)

    def test_get_q_values(self):
        """Test Q-value extraction at specific quantile"""
        qnet = TemporalQNet(latent_dim=512, hidden_dim=512, encoder_type="gru")
        z_seq = torch.randn(4, 32, 512)
        q_values = qnet.get_q_values(z_seq, quantile=0.5)
        assert q_values.shape == (4, 2)

    def test_get_ev(self):
        """Test EV extraction"""
        qnet = TemporalQNet(latent_dim=512, hidden_dim=512, encoder_type="gru")
        z_seq = torch.randn(4, 32, 512)
        ev_stay, ev_extract = qnet.get_ev(z_seq, quantile=0.5)
        assert ev_stay.shape == (4,)
        assert ev_extract.shape == (4,)


class TestEVModel:
    """Test suite for complete EVModel"""

    def test_model_creation(self):
        """Test model initialization"""
        model = EVModel(
            encoder_backbone="resnet34",
            latent_dim=512,
            encoder_pretrained=False,
            hidden_dim=512,
            temporal_encoder="gru",
        )
        assert model.latent_dim == 512
        assert model.num_actions == 2

    def test_forward_frames(self):
        """Test end-to-end forward from frames"""
        model = EVModel(encoder_pretrained=False)
        frames = torch.randn(2, 32, 3, 225, 400)
        q_dist = model.forward_frames(frames)
        assert q_dist.shape == (2, 2, 16)

    def test_forward_latents(self):
        """Test forward from latents"""
        model = EVModel(encoder_pretrained=False)
        latents = torch.randn(2, 32, 512)
        q_dist = model.forward_latents(latents)
        assert q_dist.shape == (2, 2, 16)

    def test_predict_ev(self):
        """Test EV prediction"""
        model = EVModel(encoder_pretrained=False)
        frames = torch.randn(2, 32, 3, 225, 400)
        ev_stay, ev_extract, delta_ev = model.predict_ev(frames=frames, quantile=0.5)
        assert ev_stay.shape == (2,)
        assert ev_extract.shape == (2,)
        assert delta_ev.shape == (2,)

    def test_get_recommendation(self):
        """Test recommendation generation"""
        model = EVModel(encoder_pretrained=False)
        frames = torch.randn(1, 32, 3, 225, 400)
        action, ev_s, ev_e, delta = model.get_recommendation(frames=frames, quantile=0.5)
        assert action in ["stay", "extract", "neutral"]
        assert isinstance(ev_s, float)
        assert isinstance(ev_e, float)
        assert isinstance(delta, float)


class TestModelSaveLoad:
    """Test model save/load with safetensors"""

    def test_save_load_encoder(self):
        """Test saving and loading encoder"""
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = FrameEncoder(backbone="resnet34", latent_dim=512, pretrained=False)
            path = Path(tmpdir) / "encoder.safetensors"

            # Save
            save_model_safetensors(encoder, path, metadata={"test": "value"})
            assert path.exists()

            # Load into new encoder
            encoder2 = FrameEncoder(backbone="resnet34", latent_dim=512, pretrained=False)
            metadata = load_model_safetensors(encoder2, path)

            # Verify weights match
            for (n1, p1), (n2, p2) in zip(
                encoder.named_parameters(), encoder2.named_parameters()
            ):
                assert torch.allclose(p1, p2), f"Mismatch at {n1}"

    def test_save_load_evmodel(self):
        """Test saving and loading complete model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = EVModel(encoder_pretrained=False)
            path = Path(tmpdir) / "evmodel.safetensors"

            # Save
            save_model_safetensors(model, path)

            # Load into new model
            model2 = EVModel(encoder_pretrained=False)
            load_model_safetensors(model2, path)

            # Verify outputs match
            frames = torch.randn(1, 32, 3, 225, 400)
            with torch.no_grad():
                out1 = model.forward_frames(frames)
                out2 = model2.forward_frames(frames)
                assert torch.allclose(out1, out2, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
