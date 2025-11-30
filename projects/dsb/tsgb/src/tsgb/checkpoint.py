"""Checkpoint utilities for saving and loading training state.

Uses safetensors for efficient and safe tensor serialization,
with JSON for non-tensor metadata.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from tsgb.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainerState:
    """Training state metadata."""

    global_step: int
    episode_index: int
    rng_seed: int
    timestamp: str
    attacker_model_name: str
    guard_model_name: str
    target_model_name: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainerState":
        """Create from dictionary."""
        return cls(**data)


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    attacker_model: torch.nn.Module,
    guard_model: torch.nn.Module,
    attacker_optimizer: torch.optim.Optimizer,
    guard_optimizer: torch.optim.Optimizer,
    trainer_state: TrainerState,
) -> Path:
    """Save a complete training checkpoint.

    Saves:
    - attacker.safetensors: Attacker model weights
    - guard.safetensors: Guard model weights
    - optim_attacker.safetensors: Attacker optimizer state tensors
    - optim_guard.safetensors: Guard optimizer state tensors
    - trainer_state.json: Training metadata and optimizer non-tensor state

    Args:
        checkpoint_dir: Directory to save checkpoint files.
        attacker_model: The attacker model.
        guard_model: The guard model.
        attacker_optimizer: Attacker optimizer.
        guard_optimizer: Guard optimizer.
        trainer_state: Training state metadata.

    Returns:
        Path to the checkpoint directory.
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Update timestamp
    trainer_state.timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(
        "saving_checkpoint",
        path=str(checkpoint_path),
        step=trainer_state.global_step,
        episode=trainer_state.episode_index,
    )

    # Save model weights
    save_file(
        dict(attacker_model.state_dict()),
        checkpoint_path / "attacker.safetensors",
    )
    save_file(
        dict(guard_model.state_dict()),
        checkpoint_path / "guard.safetensors",
    )

    # Save optimizer states (tensors only)
    attacker_optim_tensors, attacker_optim_meta = _split_optimizer_state(
        attacker_optimizer.state_dict()
    )
    guard_optim_tensors, guard_optim_meta = _split_optimizer_state(guard_optimizer.state_dict())

    if attacker_optim_tensors:
        save_file(attacker_optim_tensors, checkpoint_path / "optim_attacker.safetensors")
    if guard_optim_tensors:
        save_file(guard_optim_tensors, checkpoint_path / "optim_guard.safetensors")

    # Save metadata (trainer state + optimizer non-tensor state)
    metadata = {
        "trainer_state": trainer_state.to_dict(),
        "attacker_optimizer_meta": attacker_optim_meta,
        "guard_optimizer_meta": guard_optim_meta,
    }

    with open(checkpoint_path / "trainer_state.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("checkpoint_saved", path=str(checkpoint_path))
    return checkpoint_path


def load_training_checkpoint(
    checkpoint_dir: str | Path,
    attacker_model: torch.nn.Module,
    guard_model: torch.nn.Module,
    attacker_optimizer: torch.optim.Optimizer | None = None,
    guard_optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> TrainerState:
    """Load a training checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        attacker_model: The attacker model to load weights into.
        guard_model: The guard model to load weights into.
        attacker_optimizer: Optional attacker optimizer to restore state.
        guard_optimizer: Optional guard optimizer to restore state.
        device: Device to load tensors to.

    Returns:
        The restored TrainerState.

    Raises:
        FileNotFoundError: If checkpoint files are missing.
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    logger.info("loading_checkpoint", path=str(checkpoint_path))

    # Load model weights
    attacker_weights = load_file(checkpoint_path / "attacker.safetensors", device=str(device))
    guard_weights = load_file(checkpoint_path / "guard.safetensors", device=str(device))

    attacker_model.load_state_dict(attacker_weights)
    guard_model.load_state_dict(guard_weights)

    # Load metadata
    with open(checkpoint_path / "trainer_state.json") as f:
        metadata = json.load(f)

    trainer_state = TrainerState.from_dict(metadata["trainer_state"])

    # Load optimizer states if provided
    if attacker_optimizer is not None:
        attacker_optim_path = checkpoint_path / "optim_attacker.safetensors"
        if attacker_optim_path.exists():
            attacker_optim_tensors = load_file(attacker_optim_path, device=str(device))
            optim_state = _merge_optimizer_state(
                attacker_optim_tensors,
                metadata.get("attacker_optimizer_meta", {}),
            )
            attacker_optimizer.load_state_dict(optim_state)

    if guard_optimizer is not None:
        guard_optim_path = checkpoint_path / "optim_guard.safetensors"
        if guard_optim_path.exists():
            guard_optim_tensors = load_file(guard_optim_path, device=str(device))
            optim_state = _merge_optimizer_state(
                guard_optim_tensors,
                metadata.get("guard_optimizer_meta", {}),
            )
            guard_optimizer.load_state_dict(optim_state)

    logger.info(
        "checkpoint_loaded",
        path=str(checkpoint_path),
        step=trainer_state.global_step,
        episode=trainer_state.episode_index,
    )

    return trainer_state


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the most recent checkpoint in a directory.

    Looks for subdirectories with trainer_state.json and returns
    the one with the highest global_step.

    Args:
        checkpoint_dir: Base directory to search for checkpoints.

    Returns:
        Path to the latest checkpoint, or None if not found.
    """
    checkpoint_base = Path(checkpoint_dir)

    if not checkpoint_base.exists():
        return None

    # Check if this is a direct checkpoint directory
    if (checkpoint_base / "trainer_state.json").exists():
        return checkpoint_base

    # Search for checkpoint subdirectories
    latest_step = -1
    latest_path: Path | None = None

    for subdir in checkpoint_base.iterdir():
        if not subdir.is_dir():
            continue

        state_file = subdir / "trainer_state.json"
        if not state_file.exists():
            continue

        try:
            with open(state_file) as f:
                metadata = json.load(f)
                step = metadata.get("trainer_state", {}).get("global_step", -1)
                if step > latest_step:
                    latest_step = step
                    latest_path = subdir
        except (json.JSONDecodeError, KeyError):
            continue

    if latest_path:
        logger.info("found_latest_checkpoint", path=str(latest_path), step=latest_step)

    return latest_path


def load_guard_weights(
    checkpoint_dir: str | Path,
    guard_model: torch.nn.Module,
    *,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> Path | None:
    """Load guard model weights from the latest checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints or a specific checkpoint path.
        guard_model: Guard model instance to load weights into.
        device: Optional device to load tensors to. Defaults to the model's device.
        strict: Whether to enforce that checkpoint keys match the model.

    Returns:
        Path to the checkpoint directory if loaded, otherwise None.
    """
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)

    if checkpoint_path is None:
        logger.warning("guard_checkpoint_not_found", path=str(checkpoint_dir))
        return None

    weights_path = checkpoint_path / "guard.safetensors"
    if not weights_path.exists():
        logger.warning("guard_weights_missing", path=str(weights_path))
        return None

    target_device = device or next(guard_model.parameters()).device
    weights = load_file(weights_path, device=str(target_device))

    guard_model.load_state_dict(weights, strict=strict)

    logger.info(
        "guard_weights_loaded",
        path=str(weights_path),
        device=str(target_device),
        strict=strict,
    )

    return checkpoint_path


def _split_optimizer_state(
    state_dict: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Split optimizer state into tensors and non-tensors.

    Args:
        state_dict: Optimizer state dictionary.

    Returns:
        Tuple of (tensor dict for safetensors, non-tensor metadata).
    """
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {}

    # Handle 'state' which contains per-parameter state
    if "state" in state_dict:
        metadata["state_keys"] = []
        for param_id, param_state in state_dict["state"].items():
            param_id_str = str(param_id)
            metadata["state_keys"].append(param_id_str)

            for key, value in param_state.items():
                if isinstance(value, torch.Tensor):
                    tensors[f"state.{param_id_str}.{key}"] = value
                else:
                    if "state_meta" not in metadata:
                        metadata["state_meta"] = {}
                    if param_id_str not in metadata["state_meta"]:
                        metadata["state_meta"][param_id_str] = {}
                    metadata["state_meta"][param_id_str][key] = value

    # Handle 'param_groups'
    if "param_groups" in state_dict:
        # Store param_groups structure without 'params' (which are indices)
        metadata["param_groups"] = []
        for group in state_dict["param_groups"]:
            group_copy = {k: v for k, v in group.items()}
            metadata["param_groups"].append(group_copy)

    return tensors, metadata


def _merge_optimizer_state(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Merge optimizer tensors and metadata back into state dict.

    Args:
        tensors: Loaded tensor dictionary.
        metadata: Non-tensor metadata.

    Returns:
        Complete optimizer state dictionary.
    """
    state_dict: dict[str, Any] = {}

    # Reconstruct 'state'
    if "state_keys" in metadata:
        state_dict["state"] = {}
        for param_id_str in metadata["state_keys"]:
            param_id = int(param_id_str) if param_id_str.isdigit() else param_id_str
            param_state: dict[str, Any] = {}

            # Get tensors for this parameter
            prefix = f"state.{param_id_str}."
            for key, value in tensors.items():
                if key.startswith(prefix):
                    state_key = key[len(prefix) :]
                    param_state[state_key] = value

            # Get non-tensor state
            if "state_meta" in metadata and param_id_str in metadata["state_meta"]:
                param_state.update(metadata["state_meta"][param_id_str])

            state_dict["state"][param_id] = param_state

    # Reconstruct 'param_groups'
    if "param_groups" in metadata:
        state_dict["param_groups"] = metadata["param_groups"]

    return state_dict
