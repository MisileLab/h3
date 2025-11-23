"""Utilities for model saving/loading with safetensors

Provides functions to save and load PyTorch models using safetensors format,
which is safer and faster than pickle-based formats.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


def save_model_safetensors(
    model: nn.Module,
    path: Path | str,
    metadata: Dict[str, str] | None = None,
) -> None:
    """
    Save PyTorch model to safetensors format.

    Args:
        model: PyTorch model to save
        path: Path to save file (*.safetensors)
        metadata: Optional metadata dictionary (must be string values)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get state dict
    state_dict = model.state_dict()

    # Convert all tensors to contiguous (safetensors requirement)
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    # Prepare metadata
    if metadata is None:
        metadata = {}

    # Add model class name
    metadata["model_class"] = model.__class__.__name__

    # Convert all metadata values to strings
    metadata = {k: str(v) for k, v in metadata.items()}

    # Save
    save_file(state_dict, str(path), metadata=metadata)
    logger.info(f"Model saved to {path}")


def load_model_safetensors(
    model: nn.Module,
    path: Path | str,
    strict: bool = True,
    device: torch.device | str = "cpu",
) -> Dict[str, str]:
    """
    Load PyTorch model from safetensors format.

    Args:
        model: PyTorch model to load weights into
        path: Path to safetensors file
        strict: Whether to strictly enforce state dict keys match
        device: Device to load tensors to

    Returns:
        Metadata dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # Load state dict
    state_dict = load_file(str(path), device=str(device))

    # Load into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    if missing_keys:
        logger.warning(f"Missing keys when loading model: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")

    logger.info(f"Model loaded from {path}")

    # Load metadata (safetensors stores it in the file)
    # We need to re-open to get metadata
    import safetensors

    with open(path, "rb") as f:
        data = f.read()
        metadata = safetensors.safe_open(path, framework="pt").metadata()

    return metadata if metadata else {}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path | str,
    additional_state: Dict[str, Any] | None = None,
) -> None:
    """
    Save full training checkpoint (model + optimizer + state).

    Uses standard torch.save for optimizer and state, safetensors for model.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        path: Path to checkpoint directory
        additional_state: Additional state to save
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save model with safetensors
    model_path = path / "model.safetensors"
    metadata = {
        "epoch": str(epoch),
    }
    save_model_safetensors(model, model_path, metadata)

    # Save optimizer and other state with torch
    state_path = path / "training_state.pt"
    state = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if additional_state:
        state.update(additional_state)

    torch.save(state, state_path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: Path | str,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """
    Load full training checkpoint.

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        path: Path to checkpoint directory
        device: Device to load to

    Returns:
        Training state dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")

    # Load model
    model_path = path / "model.safetensors"
    load_model_safetensors(model, model_path, device=device)

    # Load training state
    state_path = path / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location=device)

        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        logger.info(f"Checkpoint loaded from {path}, epoch {state.get('epoch', 'unknown')}")
        return state
    else:
        logger.warning(f"Training state not found at {state_path}")
        return {}


def test_save_load():
    """Test safetensors save/load"""
    import tempfile

    print("Testing safetensors save/load...")

    # Create dummy model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test save/load model
        model_path = Path(tmpdir) / "test_model.safetensors"
        save_model_safetensors(model, model_path, metadata={"test": "value"})
        assert model_path.exists()

        # Load into new model
        model2 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )
        metadata = load_model_safetensors(model2, model_path)
        print(f"Loaded metadata: {metadata}")

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Weights mismatch at {n1}"

        # Test checkpoint save/load
        optimizer = torch.optim.Adam(model.parameters())
        checkpoint_path = Path(tmpdir) / "checkpoint"
        save_checkpoint(model, optimizer, epoch=5, path=checkpoint_path)

        model3 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )
        optimizer3 = torch.optim.Adam(model3.parameters())
        state = load_checkpoint(model3, optimizer3, checkpoint_path)
        print(f"Loaded checkpoint state: epoch={state.get('epoch')}")

        assert state["epoch"] == 5

    print("✓ Safetensors save/load tests passed")


if __name__ == "__main__":
    test_save_load()
