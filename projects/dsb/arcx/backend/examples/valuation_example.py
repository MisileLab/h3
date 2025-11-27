"""Example: Using vision model-based item valuation.

This example demonstrates how to integrate vision model-based item valuation
into the training/evaluation workflow.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from arcx.training import EvaluationPipeline
from arcx.valuation import ItemValuator
from arcx.data.logger import DataLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_usage():
    """Example 1: Basic usage with auto-valuation"""
    logger.info("=== Example 1: Basic Usage ===")

    # Create pipeline with auto-valuation enabled
    pipeline = EvaluationPipeline(
        auto_valuate=True,
        game_phase="mid_wipe",
    )

    # Start a run
    pipeline.start_run(run_id="raid_001", map_id="forest")

    # Simulate gameplay with decision logging
    for i in range(10):
        # Simulate latent sequence (in real usage, this comes from encoder)
        z_seq = torch.randn(32, 512)

        # Log decision
        pipeline.log_decision(
            decision_idx=i,
            t_sec=i * 5.0,
            action="stay" if i < 9 else "extract",
            z_seq=z_seq,
        )

    # End of run: load extraction screenshot
    # In real usage, this comes from screen capture
    screenshot_path = Path("data/extraction_screenshots/example.png")

    if screenshot_path.exists():
        screenshot = cv2.imread(str(screenshot_path))

        # Auto-valuate and end run
        pipeline.end_run_with_screenshot(
            screenshot=screenshot,
            total_time_sec=50.0,
            success=True,
        )

        logger.info("Run ended with auto-valuation")
    else:
        # Fallback to manual value
        pipeline.end_run_manual(
            final_loot_value=2500.0,
            total_time_sec=50.0,
            success=True,
        )

        logger.info("Run ended with manual value (no screenshot available)")

    # Save data
    pipeline.save()

    # Get stats
    stats = pipeline.get_stats()
    logger.info(f"Pipeline stats: {stats}")


def example_2_manual_valuation():
    """Example 2: Manual screenshot valuation"""
    logger.info("\n=== Example 2: Manual Valuation ===")

    # Create valuator with vision model
    valuator = ItemValuator(
        api_key=None,  # Uses OPENAI_API_KEY from environment
        confidence=0.5,
        game_phase="mid_wipe",
    )

    # Load a screenshot
    screenshot_path = Path("data/extraction_screenshots/example.png")

    if screenshot_path.exists():
        screenshot = cv2.imread(str(screenshot_path))

        # Valuate screenshot
        result = valuator.valuate_screenshot(screenshot)

        logger.info(f"Valuation result:")
        logger.info(f"  Total value: {result.total_value:.2f}")
        logger.info(f"  Items detected: {result.num_items}")
        logger.info(f"  Avg confidence: {result.avg_confidence:.2f}")
        logger.info(f"  Value breakdown: {result.value_breakdown}")
        logger.info(f"  Rarity counts: {result.rarity_counts}")
    else:
        logger.warning(f"Screenshot not found: {screenshot_path}")
        logger.info("Create example screenshot first:")
        logger.info("  1. Take screenshot of extraction screen")
        logger.info(f"  2. Save to {screenshot_path}")


def example_3_training_integration():
    """Example 3: Integration with training loop"""
    logger.info("\n=== Example 3: Training Integration ===")

    # Create custom data logger
    data_logger = DataLogger(
        log_dir=Path("data/training_logs"),
        auto_save_interval=50,
    )

    # Create valuator with specific settings
    valuator = ItemValuator(
        api_key=None,
        confidence=0.6,  # Higher confidence threshold
        game_phase="early_wipe",  # Adjust for game phase
    )

    # Create pipeline
    pipeline = EvaluationPipeline(
        data_logger=data_logger,
        valuator=valuator,
        auto_valuate=True,
        game_phase="early_wipe",
    )

    # Simulate multiple runs
    for run_idx in range(3):
        run_id = f"training_run_{run_idx:03d}"
        pipeline.start_run(run_id=run_id, map_id="desert")

        # Simulate decisions
        for dec_idx in range(15):
            z_seq = torch.randn(32, 512)
            pipeline.log_decision(
                decision_idx=dec_idx,
                t_sec=dec_idx * 8.0,
                action="stay" if dec_idx < 14 else "extract",
                z_seq=z_seq,
            )

        # End run with dummy screenshot or manual value
        screenshot_path = Path(f"data/extraction_screenshots/run_{run_idx}.png")

        if screenshot_path.exists():
            screenshot = cv2.imread(str(screenshot_path))
            pipeline.end_run_with_screenshot(
                screenshot=screenshot,
                total_time_sec=120.0,
                success=True,
                manual_value=3000.0,  # Fallback
            )
        else:
            # Use manual value
            pipeline.end_run_manual(
                final_loot_value=2000.0 + run_idx * 500,
                total_time_sec=120.0,
                success=True,
            )

        logger.info(f"Completed {run_id}")

    # Save all data
    pipeline.save()

    # Final stats
    stats = pipeline.get_stats()
    logger.info(f"\nFinal stats: {stats}")


def example_4_no_vision_model():
    """Example 4: Using pipeline without vision model (legacy mode)"""
    logger.info("\n=== Example 4: Legacy Mode (No Vision Model) ===")

    # Create pipeline with auto-valuation disabled
    pipeline = EvaluationPipeline(
        auto_valuate=False,  # Disable vision model
    )

    # Use as before with manual values
    pipeline.start_run(run_id="legacy_run", map_id="city")

    for i in range(5):
        z_seq = torch.randn(32, 512)
        pipeline.log_decision(i, i * 10.0, "stay" if i < 4 else "extract", z_seq)

    # Always use manual value
    pipeline.end_run_manual(
        final_loot_value=1500.0,
        total_time_sec=40.0,
        success=True,
    )

    pipeline.save()

    logger.info("Legacy mode run completed")


if __name__ == "__main__":
    logger.info("Vision Model Valuation Pipeline Examples\n")

    # Run examples
    example_1_basic_usage()
    example_2_manual_valuation()
    example_3_training_integration()
    example_4_no_vision_model()

    logger.info("\n✓ All examples completed")
