"""Extraction screen detector and screenshot capture.

Detects when extraction screen is shown and captures it for YOLO valuation.
"""

import logging
from pathlib import Path
from typing import Optional
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ExtractionDetector:
    """
    Detects extraction screen and captures screenshots for valuation.

    This is a simple implementation. For production, you may want to:
    - Use template matching for extraction screen UI elements
    - Use OCR to detect "Extraction Success" text
    - Use color histogram matching
    """

    def __init__(
        self,
        screenshot_dir: Optional[Path] = None,
        save_screenshots: bool = True,
    ):
        """
        Initialize extraction detector.

        Args:
            screenshot_dir: Directory to save extraction screenshots
            save_screenshots: Whether to save screenshots to disk
        """
        self.screenshot_dir = screenshot_dir or Path("data/extraction_screenshots")
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.save_screenshots = save_screenshots

        self.last_extraction_time = 0.0
        self.min_detection_interval = 5.0  # Minimum seconds between detections

        logger.info(f"ExtractionDetector initialized, saving to {self.screenshot_dir}")

    def is_extraction_screen(self, frame: np.ndarray) -> bool:
        """
        Detect if current frame is an extraction screen.

        This is a placeholder implementation. You should customize this
        based on your game's extraction screen UI.

        Args:
            frame: Screenshot frame (H, W, C) in BGR

        Returns:
            True if extraction screen detected
        """
        # Placeholder: Always return False
        # TODO: Implement actual detection logic
        # Examples:
        # - Template matching for specific UI elements
        # - OCR for "Extraction Success" text
        # - Color analysis of specific screen regions
        # - ML-based screen classification

        return False

    def detect_and_capture(
        self,
        frame: np.ndarray,
        run_id: str,
    ) -> Optional[np.ndarray]:
        """
        Detect extraction screen and capture if found.

        Args:
            frame: Current screenshot frame (H, W, C) in BGR
            run_id: Current run identifier

        Returns:
            Captured frame if extraction detected, None otherwise
        """
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_extraction_time < self.min_detection_interval:
            return None

        # Detect extraction screen
        if not self.is_extraction_screen(frame):
            return None

        logger.info("Extraction screen detected!")
        self.last_extraction_time = current_time

        # Save screenshot
        if self.save_screenshots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{run_id}_{timestamp}.png"
            filepath = self.screenshot_dir / filename

            cv2.imwrite(str(filepath), frame)
            logger.info(f"Saved extraction screenshot: {filepath}")

        return frame

    def manual_capture(
        self,
        frame: np.ndarray,
        run_id: str,
    ) -> np.ndarray:
        """
        Manually capture extraction screenshot (bypasses detection).

        Use this when you know the extraction screen is shown.

        Args:
            frame: Extraction screenshot (H, W, C) in BGR
            run_id: Run identifier

        Returns:
            The captured frame
        """
        if self.save_screenshots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{run_id}_{timestamp}_manual.png"
            filepath = self.screenshot_dir / filename

            cv2.imwrite(str(filepath), frame)
            logger.info(f"Manually saved extraction screenshot: {filepath}")

        return frame

    def set_detection_template(self, template_path: Path):
        """
        Load a template image for extraction screen detection.

        Args:
            template_path: Path to template image file
        """
        # TODO: Implement template matching
        raise NotImplementedError(
            "Template matching not yet implemented. "
            "Customize is_extraction_screen() method instead."
        )


def test_extraction_detector():
    """Test extraction detector"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        detector = ExtractionDetector(screenshot_dir=Path(tmpdir))

        # Create dummy frame
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # Test manual capture
        captured = detector.manual_capture(frame, run_id="test_run")
        assert captured is not None
        assert captured.shape == frame.shape

        logger.info("✓ ExtractionDetector tests passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_extraction_detector()
