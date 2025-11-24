"""Item valuator for calculating total loot value."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import get_item_value, apply_phase_multiplier, SET_BONUSES
from .detector import ItemDetector

logger = logging.getLogger(__name__)


@dataclass
class DetectedItem:
    """Detected item with metadata."""

    item_type: str
    rarity: str
    confidence: float
    estimated_value: float
    bbox: List[float]


@dataclass
class ValuationResult:
    """Complete valuation result."""

    total_value: float
    items: List[DetectedItem]
    num_items: int
    avg_confidence: float
    value_breakdown: dict  # item_type -> total_value
    rarity_counts: dict  # rarity -> count
    phase_multiplier: float = 1.0


class ItemValuator:
    """
    Item valuator that combines YOLO detection with value calculation.

    This class:
    1. Detects items using YOLO11
    2. Calculates individual item values
    3. Applies bonuses and multipliers
    4. Returns total loot value
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.5,
        device: str = "cuda",
        game_phase: str = "mid_wipe",
    ):
        """
        Initialize item valuator.

        Args:
            model_path: Path to YOLO11 model
            confidence: Detection confidence threshold
            device: Device for inference
            game_phase: Current game phase for value multipliers
        """
        self.detector = ItemDetector(
            model_path=model_path,
            confidence=confidence,
            device=device,
        )
        self.game_phase = game_phase
        logger.info(f"ItemValuator initialized with game phase: {game_phase}")

    def valuate_screenshot(
        self, screenshot: np.ndarray
    ) -> ValuationResult:
        """
        Valuate items from extraction screenshot.

        Args:
            screenshot: Screenshot image as numpy array (H, W, C) in BGR

        Returns:
            ValuationResult containing total value and item details
        """
        # Detect items
        detections = self.detector.detect(screenshot)

        if not detections:
            logger.warning("No items detected in screenshot")
            return ValuationResult(
                total_value=0.0,
                items=[],
                num_items=0,
                avg_confidence=0.0,
                value_breakdown={},
                rarity_counts={},
            )

        # Convert detections to DetectedItem objects
        items = []
        for item_type, rarity, conf, bbox in detections:
            base_value = get_item_value(item_type, rarity)
            adjusted_value = apply_phase_multiplier(base_value, self.game_phase)

            items.append(
                DetectedItem(
                    item_type=item_type,
                    rarity=rarity,
                    confidence=conf,
                    estimated_value=adjusted_value,
                    bbox=bbox,
                )
            )

        # Calculate total value
        base_total = sum(item.estimated_value for item in items)

        # Apply set bonuses
        bonus_multiplier = self._calculate_set_bonus(items)
        total_value = base_total * bonus_multiplier

        # Calculate statistics
        value_breakdown = self._calculate_value_breakdown(items)
        rarity_counts = self._calculate_rarity_counts(items)
        avg_confidence = sum(item.confidence for item in items) / len(items)

        logger.info(
            f"Valuated {len(items)} items, "
            f"total value: {total_value:.2f} "
            f"(bonus: {bonus_multiplier:.2f}x)"
        )

        phase_mult = apply_phase_multiplier(1.0, self.game_phase)

        return ValuationResult(
            total_value=total_value,
            items=items,
            num_items=len(items),
            avg_confidence=avg_confidence,
            value_breakdown=value_breakdown,
            rarity_counts=rarity_counts,
            phase_multiplier=phase_mult,
        )

    def _calculate_set_bonus(self, items: List[DetectedItem]) -> float:
        """
        Calculate set bonus multiplier based on item combinations.

        Args:
            items: List of detected items

        Returns:
            Bonus multiplier (>= 1.0)
        """
        item_types = {item.item_type for item in items}

        # Full loadout bonus
        if len(item_types) >= 4:
            return SET_BONUSES.get("full_loadout", 1.0)

        # Weapon + armor combo
        if "weapon" in item_types and "armor" in item_types:
            return SET_BONUSES.get("weapon_armor_combo", 1.0)

        return 1.0

    def _calculate_value_breakdown(
        self, items: List[DetectedItem]
    ) -> dict:
        """
        Calculate total value per item type.

        Args:
            items: List of detected items

        Returns:
            Dictionary mapping item_type to total value
        """
        breakdown = {}
        for item in items:
            if item.item_type not in breakdown:
                breakdown[item.item_type] = 0.0
            breakdown[item.item_type] += item.estimated_value

        return breakdown

    def _calculate_rarity_counts(self, items: List[DetectedItem]) -> dict:
        """
        Count items by rarity.

        Args:
            items: List of detected items

        Returns:
            Dictionary mapping rarity to count
        """
        counts = {}
        for item in items:
            if item.rarity not in counts:
                counts[item.rarity] = 0
            counts[item.rarity] += 1

        return counts

    def set_game_phase(self, phase: str):
        """
        Update game phase for value calculations.

        Args:
            phase: New game phase ("early_wipe", "mid_wipe", "late_wipe")
        """
        self.game_phase = phase
        logger.info(f"Game phase updated to: {phase}")
