"""Device backend detection for PyTorch

Automatically detects the best available backend in order:
1. NVIDIA CUDA
2. AMD ROCm (on Windows)
3. DirectML (AMD/Intel on Windows)
4. CPU fallback
"""

import enum
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DeviceBackend(enum.Enum):
    """Available device backends"""

    CUDA = "cuda"
    ROCM = "rocm"
    DML = "directml"
    CPU = "cpu"


def detect_backend() -> tuple[DeviceBackend, torch.device]:
    """
    Detect the best available device backend.

    Returns:
        Tuple of (backend_type, torch_device)
    """
    # Priority 1: NVIDIA CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Detected CUDA device: {device_name}")

        # Check if it's AMD with ROCm
        if "AMD" in device_name or "Radeon" in device_name:
            logger.info("AMD GPU detected with ROCm support")
            return DeviceBackend.ROCM, torch.device("cuda:0")
        else:
            logger.info("NVIDIA GPU detected")
            return DeviceBackend.CUDA, torch.device("cuda:0")

    # Priority 2: DirectML (AMD/Intel on Windows)
    try:
        import torch_directml  # type: ignore

        dml_device = torch_directml.device()
        logger.info("DirectML backend available")
        return DeviceBackend.DML, dml_device
    except ImportError:
        logger.debug("DirectML not available")

    # Fallback: CPU
    logger.warning("No GPU backend available, falling back to CPU")
    return DeviceBackend.CPU, torch.device("cpu")


class DeviceManager:
    """Manages device backend and provides utilities"""

    def __init__(self):
        self.backend, self.device = detect_backend()
        logger.info(f"Using backend: {self.backend.value}, device: {self.device}")

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the managed device"""
        return tensor.to(self.device)

    def to_device_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Move module to the managed device"""
        return module.to(self.device)

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU backend"""
        return self.backend != DeviceBackend.CPU

    def get_memory_stats(self) -> Optional[dict]:
        """Get GPU memory statistics if available"""
        if self.backend == DeviceBackend.CUDA or self.backend == DeviceBackend.ROCM:
            return {
                "allocated": torch.cuda.memory_allocated(self.device),
                "reserved": torch.cuda.memory_reserved(self.device),
                "max_allocated": torch.cuda.max_memory_allocated(self.device),
            }
        return None

    def __repr__(self) -> str:
        return f"DeviceManager(backend={self.backend.value}, device={self.device})"


# Global device manager instance
device_manager = DeviceManager()
