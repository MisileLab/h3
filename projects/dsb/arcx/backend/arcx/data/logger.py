"""Data logger for writing decision logs to Parquet

Collects decision data during gameplay and saves to Parquet files.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import json

import numpy as np
import polars as pl
import torch

from arcx.data.schema import (
    create_decision_log_df,
    create_feedback_log_df,
    flatten_latent_sequence,
    DECISION_LOG_SCHEMA,
    FEEDBACK_LOG_SCHEMA,
)
from arcx.config import config

logger = logging.getLogger(__name__)


@dataclass
class DecisionPoint:
    """Single decision point data"""

    decision_idx: int
    t_sec: float
    action: str  # "stay" or "extract"
    z_seq: torch.Tensor  # [L, D] latent sequence


@dataclass
class RunData:
    """Data for a complete run"""

    run_id: str
    map_id: str
    start_time: float
    decisions: List[DecisionPoint] = field(default_factory=list)

    # Populated at end of run
    final_loot_value: Optional[float] = None
    total_time_sec: Optional[float] = None
    success: Optional[bool] = None

    # YOLO valuation results (optional)
    num_items_detected: Optional[int] = None
    avg_detection_confidence: Optional[float] = None
    value_breakdown: Optional[Dict[str, float]] = None  # item_type -> value
    rarity_counts: Optional[Dict[str, int]] = None  # rarity -> count


class DataLogger:
    """
    Logger that collects decision data and writes to Parquet.

    Usage:
        logger = DataLogger()
        logger.start_run("run_123", "map_forest")
        logger.log_decision(decision_idx=0, t_sec=10.5, action="stay", z_seq=latents)
        logger.log_decision(decision_idx=1, t_sec=25.3, action="extract", z_seq=latents)
        logger.end_run(final_loot=2500, total_time=180, success=True)
        logger.save()
    """

    def __init__(self, log_dir: Optional[Path] = None, auto_save_interval: int = 100):
        """
        Args:
            log_dir: Directory to save logs (default: from config)
            auto_save_interval: Auto-save after N decisions
        """
        self.log_dir = log_dir or config.data.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.auto_save_interval = auto_save_interval

        # Current run
        self.current_run: Optional[RunData] = None

        # Completed runs waiting to be saved
        self.completed_runs: List[RunData] = []

        # Feedback logs
        self.feedback_logs: List[Dict[str, Any]] = []

        # Counters
        self.total_decisions = 0
        self.total_runs = 0

        logger.info(f"DataLogger initialized: log_dir={self.log_dir}")

    def start_run(self, run_id: str, map_id: str = "unknown"):
        """Start a new run"""
        if self.current_run is not None:
            logger.warning(f"Starting new run {run_id} without ending previous run {self.current_run.run_id}")

        self.current_run = RunData(
            run_id=run_id,
            map_id=map_id,
            start_time=time.time(),
        )
        logger.info(f"Started run: {run_id}, map: {map_id}")

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
            decision_idx: Index of this decision in the run
            t_sec: Time since run start (seconds)
            action: "stay" or "extract"
            z_seq: [L, D] latent sequence tensor
        """
        if self.current_run is None:
            logger.warning("log_decision called without active run")
            return

        decision = DecisionPoint(
            decision_idx=decision_idx,
            t_sec=t_sec,
            action=action,
            z_seq=z_seq.cpu(),  # Move to CPU for storage
        )

        self.current_run.decisions.append(decision)
        self.total_decisions += 1

        # Auto-save check
        if self.total_decisions % self.auto_save_interval == 0:
            self.save()

    def end_run(
        self,
        final_loot_value: Optional[float] = None,
        total_time_sec: Optional[float] = None,
        success: Optional[bool] = None,
        screenshot: Optional[np.ndarray] = None,
        valuator=None,
    ):
        """
        End current run with final results.

        Args:
            final_loot_value: Total loot value (optional if screenshot provided)
            total_time_sec: Total run duration
            success: Whether extraction succeeded
            screenshot: Extraction screenshot for YOLO valuation (optional)
            valuator: ItemValuator instance for auto-valuation (optional)
        """
        if self.current_run is None:
            logger.warning("end_run called without active run")
            return

        # Auto-valuate from screenshot if provided
        if screenshot is not None and valuator is not None:
            try:
                from arcx.valuation import ItemValuator

                logger.info("Auto-valuating loot from screenshot...")
                result = valuator.valuate_screenshot(screenshot)

                # Use YOLO-calculated value
                self.current_run.final_loot_value = result.total_value
                self.current_run.num_items_detected = result.num_items
                self.current_run.avg_detection_confidence = result.avg_confidence
                self.current_run.value_breakdown = result.value_breakdown
                self.current_run.rarity_counts = result.rarity_counts

                logger.info(
                    f"YOLO valuation: {result.total_value:.2f} "
                    f"({result.num_items} items, "
                    f"avg conf: {result.avg_confidence:.2f})"
                )

            except Exception as e:
                logger.error(f"YOLO valuation failed: {e}", exc_info=True)
                # Fall back to manual value if provided
                if final_loot_value is not None:
                    self.current_run.final_loot_value = final_loot_value
                else:
                    logger.error("No loot value available!")
                    self.current_run.final_loot_value = 0.0

        elif final_loot_value is not None:
            # Use manually provided value
            self.current_run.final_loot_value = final_loot_value
        else:
            logger.error("No loot value or screenshot provided!")
            self.current_run.final_loot_value = 0.0

        # Populate other final values
        self.current_run.total_time_sec = total_time_sec or 0.0
        self.current_run.success = success if success is not None else False

        # Move to completed
        self.completed_runs.append(self.current_run)
        self.total_runs += 1

        logger.info(
            f"Ended run: {self.current_run.run_id}, "
            f"decisions={len(self.current_run.decisions)}, "
            f"loot={self.current_run.final_loot_value:.2f}, "
            f"time={self.current_run.total_time_sec}s, "
            f"success={self.current_run.success}"
        )

        self.current_run = None

        # Auto-save
        self.save()

    def log_feedback(
        self,
        run_id: str,
        decision_idx: int,
        rating: str,
        context: Optional[Dict] = None,
    ):
        """
        Log user feedback.

        Args:
            run_id: Run identifier
            decision_idx: Decision index
            rating: "bad" or "good"
            context: Additional context
        """
        feedback = {
            "run_id": run_id,
            "decision_idx": decision_idx,
            "timestamp": time.time(),
            "rating": rating,
            "context": json.dumps(context or {}),
        }
        self.feedback_logs.append(feedback)

    def save(self):
        """Save completed runs to Parquet"""
        if not self.completed_runs:
            logger.debug("No completed runs to save")
            return

        # Convert to DataFrame
        rows = []
        for run in self.completed_runs:
            for decision in run.decisions:
                # Flatten latent sequence
                z_seq_np = decision.z_seq.numpy()  # [L, D]
                z_flat = flatten_latent_sequence(z_seq_np.tolist())

                # Convert valuation dicts to JSON
                value_breakdown_json = (
                    json.dumps(run.value_breakdown)
                    if run.value_breakdown is not None
                    else None
                )
                rarity_counts_json = (
                    json.dumps(run.rarity_counts)
                    if run.rarity_counts is not None
                    else None
                )

                row = {
                    "run_id": run.run_id,
                    "decision_idx": decision.decision_idx,
                    "t_sec": decision.t_sec,
                    "map_id": run.map_id,
                    "action": decision.action,
                    "final_loot_value": run.final_loot_value,
                    "total_time_sec": run.total_time_sec,
                    "success": run.success,
                    "z_seq": z_flat,
                    "num_items_detected": run.num_items_detected,
                    "avg_detection_confidence": run.avg_detection_confidence,
                    "value_breakdown_json": value_breakdown_json,
                    "rarity_counts_json": rarity_counts_json,
                }
                rows.append(row)

        if not rows:
            logger.warning("No decision data to save")
            return

        # Create DataFrame
        df = pl.DataFrame(rows, schema=DECISION_LOG_SCHEMA)

        # Save to Parquet
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = self.log_dir / f"decisions_{timestamp}.parquet"
        df.write_parquet(
            filepath,
            compression=config.data.parquet_compression,
        )

        logger.info(f"Saved {len(rows)} decisions from {len(self.completed_runs)} runs to {filepath}")

        # Clear completed runs
        self.completed_runs.clear()

        # Save feedback if any
        if self.feedback_logs:
            self._save_feedback()

    def _save_feedback(self):
        """Save feedback logs"""
        if not self.feedback_logs:
            return

        df = pl.DataFrame(self.feedback_logs, schema=FEEDBACK_LOG_SCHEMA)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = self.log_dir / f"feedback_{timestamp}.parquet"
        df.write_parquet(filepath, compression=config.data.parquet_compression)

        logger.info(f"Saved {len(self.feedback_logs)} feedback entries to {filepath}")
        self.feedback_logs.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics"""
        return {
            "total_runs": self.total_runs,
            "total_decisions": self.total_decisions,
            "completed_runs_pending": len(self.completed_runs),
            "current_run_id": self.current_run.run_id if self.current_run else None,
            "feedback_logs_pending": len(self.feedback_logs),
        }


def test_logger():
    """Test data logger"""
    import tempfile

    print("Testing DataLogger...")

    with tempfile.TemporaryDirectory() as tmpdir:
        logger_inst = DataLogger(log_dir=Path(tmpdir), auto_save_interval=10)

        # Simulate a run
        logger_inst.start_run("run_001", "forest")

        for i in range(5):
            z_seq = torch.randn(32, 512)
            logger_inst.log_decision(
                decision_idx=i,
                t_sec=i * 30.0,
                action="stay" if i < 4 else "extract",
                z_seq=z_seq,
            )

        logger_inst.end_run(final_loot_value=2500.0, total_time_sec=150.0, success=True)

        # Check files created
        files = list(Path(tmpdir).glob("*.parquet"))
        print(f"Created files: {files}")
        assert len(files) == 1

        # Load and verify
        df = pl.read_parquet(files[0])
        print(f"Loaded DataFrame: {df.shape}")
        print(df.head())
        assert len(df) == 5
        assert df["run_id"][0] == "run_001"

        # Check stats
        stats = logger_inst.get_stats()
        print(f"Stats: {stats}")

    print("✓ DataLogger tests passed")


if __name__ == "__main__":
    test_logger()
