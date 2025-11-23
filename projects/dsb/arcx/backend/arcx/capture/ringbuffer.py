"""Ring buffer for frames and latent vectors

Maintains fixed-size circular buffers for recent frames and their latent encodings.
"""

import logging
from collections import deque
from typing import Optional, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


class RingBuffer:
    """
    Generic ring buffer with fixed capacity.

    Stores recent items and automatically overwrites oldest when full.
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of items to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, item):
        """Add item to buffer (overwrites oldest if full)"""
        self.buffer.append(item)

    def get_last_n(self, n: int) -> List:
        """
        Get last N items from buffer.

        Args:
            n: Number of items to retrieve

        Returns:
            List of items (may be shorter than N if buffer not full)
        """
        if n > len(self.buffer):
            n = len(self.buffer)

        if n == 0:
            return []

        # Get last n items
        return list(self.buffer)[-n:]

    def get_all(self) -> List:
        """Get all items in buffer"""
        return list(self.buffer)

    def clear(self):
        """Clear all items"""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity"""
        return len(self.buffer) == self.capacity


class FrameRingBuffer:
    """
    Ring buffer specifically for video frames.

    Stores frames as numpy arrays and provides batch retrieval.
    """

    def __init__(self, capacity: int, frame_shape: tuple):
        """
        Args:
            capacity: Maximum number of frames
            frame_shape: Shape of each frame (H, W, C)
        """
        self.capacity = capacity
        self.frame_shape = frame_shape
        self.buffer = RingBuffer(capacity)

    def push(self, frame: np.ndarray):
        """
        Add frame to buffer.

        Args:
            frame: [H, W, C] numpy array
        """
        assert frame.shape == self.frame_shape, f"Frame shape mismatch: {frame.shape} != {self.frame_shape}"
        self.buffer.push(frame)

    def get_last_n(self, n: int) -> Optional[np.ndarray]:
        """
        Get last N frames as a batch.

        Args:
            n: Number of frames

        Returns:
            [N, H, W, C] numpy array, or None if insufficient frames
        """
        frames = self.buffer.get_last_n(n)
        if len(frames) < n:
            return None

        return np.stack(frames, axis=0)

    def get_last_n_torch(self, n: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Get last N frames as PyTorch tensor.

        Args:
            n: Number of frames
            device: Target device

        Returns:
            [N, C, H, W] torch tensor (channel-first), or None if insufficient frames
        """
        frames_np = self.get_last_n(n)
        if frames_np is None:
            return None

        # Convert to torch: [N, H, W, C] -> [N, C, H, W]
        frames_torch = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
        frames_torch = frames_torch.to(device)

        return frames_torch

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_full(self) -> bool:
        return self.buffer.is_full


class LatentRingBuffer:
    """
    Ring buffer for latent vectors.

    Stores encoded frame features for fast Q-net inference.
    """

    def __init__(self, capacity: int, latent_dim: int):
        """
        Args:
            capacity: Maximum number of latent vectors
            latent_dim: Dimension of each latent vector
        """
        self.capacity = capacity
        self.latent_dim = latent_dim
        self.buffer = RingBuffer(capacity)

    def push(self, latent: torch.Tensor):
        """
        Add latent vector to buffer.

        Args:
            latent: [D] or [1, D] tensor
        """
        if latent.ndim == 2:
            latent = latent.squeeze(0)

        assert latent.shape == (self.latent_dim,), f"Latent shape mismatch: {latent.shape}"

        # Store on CPU to save GPU memory
        self.buffer.push(latent.cpu())

    def get_last_n(self, n: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Get last N latent vectors.

        Args:
            n: Number of vectors
            device: Target device

        Returns:
            [N, D] tensor, or None if insufficient latents
        """
        latents = self.buffer.get_last_n(n)
        if len(latents) < n:
            return None

        # Stack and move to device
        latents_tensor = torch.stack(latents, dim=0).to(device)

        return latents_tensor

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_full(self) -> bool:
        return self.buffer.is_full


class CombinedRingBuffer:
    """
    Combined buffer that maintains both frames and latents.

    Useful for debugging and data logging.
    """

    def __init__(self, capacity: int, frame_shape: tuple, latent_dim: int):
        self.frame_buffer = FrameRingBuffer(capacity, frame_shape)
        self.latent_buffer = LatentRingBuffer(capacity, latent_dim)

    def push(self, frame: np.ndarray, latent: torch.Tensor):
        """Push frame and corresponding latent"""
        self.frame_buffer.push(frame)
        self.latent_buffer.push(latent)

    def get_last_n_latents(self, n: int, device: torch.device) -> Optional[torch.Tensor]:
        """Get last N latents"""
        return self.latent_buffer.get_last_n(n, device)

    def get_last_n_frames(self, n: int) -> Optional[np.ndarray]:
        """Get last N frames"""
        return self.frame_buffer.get_last_n(n)

    def __len__(self) -> int:
        return len(self.latent_buffer)

    @property
    def is_full(self) -> bool:
        return self.latent_buffer.is_full


def test_ring_buffer():
    """Test ring buffer functionality"""
    print("Testing ring buffers...")

    # Test generic ring buffer
    rb = RingBuffer(capacity=5)
    for i in range(10):
        rb.push(i)

    assert len(rb) == 5
    assert rb.get_last_n(3) == [7, 8, 9]
    print("✓ RingBuffer test passed")

    # Test frame ring buffer
    frame_buffer = FrameRingBuffer(capacity=32, frame_shape=(225, 400, 3))
    for i in range(40):
        frame = np.random.randint(0, 256, (225, 400, 3), dtype=np.uint8)
        frame_buffer.push(frame)

    assert len(frame_buffer) == 32
    last_16 = frame_buffer.get_last_n(16)
    assert last_16.shape == (16, 225, 400, 3)
    print("✓ FrameRingBuffer test passed")

    # Test latent ring buffer
    latent_buffer = LatentRingBuffer(capacity=32, latent_dim=512)
    device = torch.device("cpu")

    for i in range(40):
        latent = torch.randn(512)
        latent_buffer.push(latent)

    assert len(latent_buffer) == 32
    last_16_latents = latent_buffer.get_last_n(16, device)
    assert last_16_latents.shape == (16, 512)
    print("✓ LatentRingBuffer test passed")

    # Test combined buffer
    combined = CombinedRingBuffer(capacity=32, frame_shape=(225, 400, 3), latent_dim=512)
    for i in range(40):
        frame = np.random.randint(0, 256, (225, 400, 3), dtype=np.uint8)
        latent = torch.randn(512)
        combined.push(frame, latent)

    assert len(combined) == 32
    assert combined.is_full
    print("✓ CombinedRingBuffer test passed")

    print("✓ All ring buffer tests passed")


if __name__ == "__main__":
    test_ring_buffer()
