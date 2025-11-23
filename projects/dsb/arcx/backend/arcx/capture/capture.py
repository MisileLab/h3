"""Screen capture for Windows using dxcam

Provides high-performance screen capture using DirectX/DXGI.
"""

import logging
import time
from typing import Optional
import numpy as np
from PIL import Image

try:
    import dxcam
    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False
    logging.warning("dxcam not available, screen capture will not work")

logger = logging.getLogger(__name__)


class ScreenCapture:
    """
    High-performance screen capture using dxcam.

    Captures game window at specified FPS and downscales to target resolution.
    """

    def __init__(
        self,
        target_fps: int = 8,
        target_width: int = 400,
        target_height: int = 225,
        device_idx: int = 0,
        output_idx: int = 0,
    ):
        """
        Args:
            target_fps: Target capture framerate
            target_width: Downscaled width
            target_height: Downscaled height
            device_idx: GPU device index
            output_idx: Monitor index
        """
        if not DXCAM_AVAILABLE:
            raise RuntimeError(
                "dxcam is not installed. Install with: pip install dxcam"
            )

        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self.target_size = (target_width, target_height)

        self.frame_interval = 1.0 / target_fps

        # Initialize dxcam
        try:
            self.camera = dxcam.create(device_idx=device_idx, output_idx=output_idx)
            logger.info(f"dxcam initialized: device={device_idx}, output={output_idx}")
        except Exception as e:
            logger.error(f"Failed to initialize dxcam: {e}")
            raise

        self.is_capturing = False
        self.last_capture_time = 0.0
        self.frame_count = 0
        self.fps_actual = 0.0

    def start(self, target_fps: Optional[int] = None, region: Optional[tuple] = None):
        """
        Start screen capture.

        Args:
            target_fps: Override target FPS
            region: Capture region (left, top, right, bottom) or None for full screen
        """
        if target_fps is not None:
            self.target_fps = target_fps
            self.frame_interval = 1.0 / target_fps

        # Start capture
        self.camera.start(target_fps=self.target_fps, region=region)
        self.is_capturing = True
        self.last_capture_time = time.time()
        self.frame_count = 0

        logger.info(f"Screen capture started: {self.target_fps} FPS, region={region}")

    def stop(self):
        """Stop screen capture"""
        if self.is_capturing:
            self.camera.stop()
            self.is_capturing = False
            logger.info("Screen capture stopped")

    def grab_frame(self) -> Optional[np.ndarray]:
        """
        Grab and process a single frame.

        Returns:
            np.ndarray: [H, W, 3] RGB frame (uint8), or None if no new frame
        """
        if not self.is_capturing:
            return None

        # Grab frame from dxcam
        frame = self.camera.get_latest_frame()

        if frame is None:
            return None

        # Convert to PIL for resize
        frame_pil = Image.fromarray(frame)

        # Resize to target resolution
        frame_resized = frame_pil.resize(self.target_size, Image.BILINEAR)

        # Convert back to numpy
        frame_np = np.array(frame_resized)

        # Update stats
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_capture_time
        if elapsed > 1.0:
            self.fps_actual = self.frame_count / elapsed
            self.frame_count = 0
            self.last_capture_time = current_time

        return frame_np

    def get_fps(self) -> float:
        """Get actual capture FPS"""
        return self.fps_actual

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class FallbackCapture:
    """
    Fallback capture for testing when dxcam is not available.

    Generates synthetic frames.
    """

    def __init__(self, target_fps: int = 8, target_width: int = 400, target_height: int = 225):
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self.is_capturing = False
        self.frame_count = 0

    def start(self, target_fps: Optional[int] = None, region: Optional[tuple] = None):
        logger.warning("Using fallback capture (synthetic frames)")
        self.is_capturing = True
        if target_fps is not None:
            self.target_fps = target_fps

    def stop(self):
        self.is_capturing = False

    def grab_frame(self) -> Optional[np.ndarray]:
        """Generate synthetic frame"""
        if not self.is_capturing:
            return None

        # Generate random frame
        frame = np.random.randint(0, 256, (self.target_height, self.target_width, 3), dtype=np.uint8)
        self.frame_count += 1

        return frame

    def get_fps(self) -> float:
        return float(self.target_fps)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_capture(**kwargs) -> ScreenCapture | FallbackCapture:
    """
    Create screen capture instance.

    Falls back to synthetic capture if dxcam is not available.
    """
    if DXCAM_AVAILABLE:
        return ScreenCapture(**kwargs)
    else:
        logger.warning("dxcam not available, using fallback capture")
        return FallbackCapture(**kwargs)


def test_capture():
    """Test screen capture"""
    print("Testing screen capture...")

    capture = create_capture(target_fps=8, target_width=400, target_height=225)

    with capture:
        capture.start()
        print(f"Capture started, is_capturing={capture.is_capturing}")

        # Grab a few frames
        for i in range(10):
            frame = capture.grab_frame()
            if frame is not None:
                print(f"Frame {i}: shape={frame.shape}, dtype={frame.dtype}")
                assert frame.shape == (225, 400, 3)
                assert frame.dtype == np.uint8
            time.sleep(0.1)

        print(f"Actual FPS: {capture.get_fps():.2f}")

    print("✓ Screen capture test passed")


if __name__ == "__main__":
    test_capture()
