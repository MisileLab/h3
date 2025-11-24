"""Evaluation pipeline with YOLO item valuation integration.

This module provides utilities for integrating YOLO-based item valuation
into the training and evaluation workflow.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
import torch

from arcx.data.logger import DataLogger
from arcx.valuation import ItemValuator
from arcx.capture.extraction_detector import ExtractionDetector

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Integrated evaluation pipeline with YOLO valuation.

    Combines data logging, screenshot capture, and automatic valuation.
    """

    def __init__(
        self,
        data_logger: Optional[DataLogger] = None,
        valuator: Optional[ItemValuator] = None,
        extraction_detector: Optional[ExtractionDetector] = None,
        auto_valuate: bool = True,
        game_phase: str = "mid_wipe",
    ):
        """
        Initialize evaluation pipeline.

        Args:
            data_logger: DataLogger instance for logging decisions
            valuator: ItemValuator for YOLO-based valuation
            extraction_detector: ExtractionDetector for screenshot capture
            auto_valuate: Whether to automatically valuate screenshots
            game_phase: Current game phase for value multipliers
        """
        self.data_logger = data_logger or DataLogger()
        self.auto_valuate = auto_valuate
        self.game_phase = game_phase

        # Initialize valuator if needed
        if auto_valuate:
            self.valuator = valuator or ItemValuator(
                model_path=None,  # Use default
                confidence=0.5,
                device="cuda",
                game_phase=game_phase,
            )
        else:
            self.valuator = None

        # Initialize extraction detector
        self.extraction_detector = extraction_detector or ExtractionDetector(
            screenshot_dir=Path("data/extraction_screenshots"),
            save_screenshots=True,
        )

        logger.info(
            f"EvaluationPipeline initialized "
            f"(auto_valuate={auto_valuate}, phase={game_phase})"
        )

    def start_run(self, run_id: str, map_id: str = "unknown"):
        """
        Start a new run.

        Args:
            run_id: Unique run identifier
            map_id: Map identifier
        """
        self.data_logger.start_run(run_id, map_id)
        logger.info(f"Started run: {run_id} on {map_id}")

    def log_decision(
        self,
        decision_idx: int,
        t_sec: float,
        action: str,
        z_seq: torch.Tensor,
    ):
        """
        Log a decision point.

        Args:
            decision_idx: Decision index
            t_sec: Time in seconds
            action: Action taken ("stay" or "extract")
            z_seq: Latent sequence tensor
        """
        self.data_logger.log_decision(decision_idx, t_sec, action, z_seq)

    def end_run_with_screenshot(
        self,
        screenshot: np.ndarray,
        total_time_sec: float,
        success: bool,
        manual_value: Optional[float] = None,
    ):
        """
        End run with extraction screenshot for automatic valuation.

        Args:
            screenshot: Extraction screenshot (H, W, C) in BGR format
            total_time_sec: Total run duration
            success: Whether extraction succeeded
            manual_value: Manual loot value (fallback if YOLO fails)
        """
        if self.auto_valuate and self.valuator is not None:
            # Auto-valuate using YOLO
            self.data_logger.end_run(
                final_loot_value=manual_value,  # Fallback
                total_time_sec=total_time_sec,
                success=success,
                screenshot=screenshot,
                valuator=self.valuator,
            )
        else:
            # Use manual value
            if manual_value is None:
                logger.error("No manual value provided and auto-valuation disabled!")
                manual_value = 0.0

            self.data_logger.end_run(
                final_loot_value=manual_value,
                total_time_sec=total_time_sec,
                success=success,
            )

    def end_run_manual(
        self,
        final_loot_value: float,
        total_time_sec: float,
        success: bool,
    ):
        """
        End run with manual loot value (no screenshot).

        Args:
            final_loot_value: Manual loot value
            total_time_sec: Total run duration
            success: Whether extraction succeeded
        """
        self.data_logger.end_run(
            final_loot_value=final_loot_value,
            total_time_sec=total_time_sec,
            success=success,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        run_id: str,
    ) -> Optional[np.ndarray]:
        """
        Process frame for extraction detection.

        Call this on every frame or periodically to detect extraction screens.

        Args:
            frame: Current frame (H, W, C) in BGR
            run_id: Current run ID

        Returns:
            Captured screenshot if extraction detected, None otherwise
        """
        return self.extraction_detector.detect_and_capture(frame, run_id)

    def valuate_screenshot(
        self,
        screenshot: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Manually valuate a screenshot.

        Args:
            screenshot: Screenshot (H, W, C) in BGR

        Returns:
            Valuation results dictionary
        """
        if self.valuator is None:
            raise RuntimeError("Valuator not initialized")

        result = self.valuator.valuate_screenshot(screenshot)

        return {
            "total_value": result.total_value,
            "num_items": result.num_items,
            "avg_confidence": result.avg_confidence,
            "value_breakdown": result.value_breakdown,
            "rarity_counts": result.rarity_counts,
        }

    def save(self):
        """Save logged data to disk."""
        self.data_logger.save()

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self.data_logger.get_stats()
        stats["auto_valuate"] = self.auto_valuate
        stats["game_phase"] = self.game_phase
        return stats


def test_evaluation_pipeline():
    """Test evaluation pipeline"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Initialize pipeline
        logger_inst = DataLogger(log_dir=tmpdir_path / "logs")
        pipeline = EvaluationPipeline(
            data_logger=logger_inst,
            auto_valuate=False,  # Disable for test
        )

        # Start run
        pipeline.start_run(run_id="test_run", map_id="test_map")

        # Log decisions
        for i in range(5):
            z_seq = torch.randn(32, 512)
            pipeline.log_decision(
                decision_idx=i,
                t_sec=i * 10.0,
                action="stay" if i < 4 else "extract",
                z_seq=z_seq,
            )

        # End run (manual)
        pipeline.end_run_manual(
            final_loot_value=2500.0,
            total_time_sec=50.0,
            success=True,
        )

        # Save
        pipeline.save()

        # Check stats
        stats = pipeline.get_stats()
        logger.info(f"Stats: {stats}")

        assert stats["total_runs"] == 1
        assert stats["total_decisions"] == 5

        logger.info("✓ EvaluationPipeline tests passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_evaluation_pipeline()
