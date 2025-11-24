"""YOLO11-based item detector for extraction screens."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from ultralytics import YOLO

from .config import MIN_DETECTION_CONFIDENCE, ITEM_TYPES, RARITY_LEVELS

logger = logging.getLogger(__name__)


class ItemDetector:
    """
    YOLO11-based item detector for extraction screens.

    This detector identifies items and their rarities from game screenshots.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = MIN_DETECTION_CONFIDENCE,
        device: str = "cuda",
    ):
        """
        Initialize item detector.

        Args:
            model_path: Path to trained YOLO11 model (.pt or .safetensors). If None, uses default.
            confidence: Minimum confidence threshold for detections
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.confidence = confidence
        self.device = device

        # Default model path - prefer safetensors
        if model_path is None:
            safetensors_path = Path("models/item_detector_yolo11.safetensors")
            pt_path = Path("models/item_detector_yolo11.pt")

            if safetensors_path.exists():
                model_path = str(safetensors_path)
            elif pt_path.exists():
                model_path = str(pt_path)
            else:
                model_path = "models/item_detector_yolo11.safetensors"

        self.model_path = Path(model_path)

        # Load model
        if self.model_path.suffix == ".safetensors":
            self.model = self._load_from_safetensors(self.model_path)
        elif self.model_path.exists():
            logger.info(f"Loading YOLO11 model from {self.model_path}")
            self.model = YOLO(str(self.model_path))
        else:
            logger.warning(
                f"Model not found at {self.model_path}, using YOLOv11n as placeholder"
            )
            self.model = YOLO("yolo11n.pt")

        # Class mapping (will be loaded from trained model)
        # Format: class_id -> (item_type, rarity)
        self.class_map = self._build_class_map()

    def _load_from_safetensors(self, safetensors_path: Path) -> YOLO:
        """
        Load YOLO model from safetensors file.

        Args:
            safetensors_path: Path to .safetensors file

        Returns:
            YOLO model with loaded weights
        """
        if not safetensors_path.exists():
            logger.error(f"Safetensors file not found: {safetensors_path}")
            raise FileNotFoundError(f"Model file not found: {safetensors_path}")

        logger.info(f"Loading YOLO11 model from safetensors: {safetensors_path}")

        # Load state dict from safetensors
        state_dict = load_file(str(safetensors_path))

        # Create base YOLO model (will be replaced with loaded weights)
        # Start with yolo11n as base architecture
        model = YOLO("yolo11n.pt")

        # Load the state dict into the model
        model.model.load_state_dict(state_dict, strict=False)

        logger.info(f"✓ Loaded {len(state_dict)} parameters from safetensors")

        return model

    def _build_class_map(self) -> dict:
        """
        Build class ID to (item_type, rarity) mapping.

        For YOLO training, we encode both item type and rarity into class IDs.
        Class ID = item_type_index * len(RARITY_LEVELS) + rarity_index

        Returns:
            Dictionary mapping class IDs to (item_type, rarity) tuples
        """
        class_map = {}
        class_id = 0

        for item_type in ITEM_TYPES:
            for rarity in RARITY_LEVELS:
                class_map[class_id] = (item_type, rarity)
                class_id += 1

        logger.info(f"Built class map with {len(class_map)} classes")
        return class_map

    def detect(
        self, image: np.ndarray
    ) -> List[Tuple[str, str, float, List[float]]]:
        """
        Detect items in image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            List of detections, each containing:
                - item_type: Type of item
                - rarity: Rarity level
                - confidence: Detection confidence score
                - bbox: Bounding box [x1, y1, x2, y2]
        """
        # Run YOLO inference
        results = self.model(
            image,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Extract detection info
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                bbox = box.xyxy[0].cpu().numpy().tolist()

                # Map class ID to item type and rarity
                if cls_id in self.class_map:
                    item_type, rarity = self.class_map[cls_id]
                else:
                    logger.warning(f"Unknown class ID: {cls_id}")
                    continue

                detections.append((item_type, rarity, conf, bbox))

        logger.debug(f"Detected {len(detections)} items")
        return detections

    def get_class_id(self, item_type: str, rarity: str) -> int:
        """
        Get class ID for item type and rarity combination.

        Args:
            item_type: Type of item
            rarity: Rarity level

        Returns:
            Class ID for YOLO training
        """
        try:
            item_idx = ITEM_TYPES.index(item_type)
            rarity_idx = RARITY_LEVELS.index(rarity)
            return item_idx * len(RARITY_LEVELS) + rarity_idx
        except ValueError:
            raise ValueError(f"Invalid item_type '{item_type}' or rarity '{rarity}'")

    @property
    def num_classes(self) -> int:
        """Get total number of classes."""
        return len(ITEM_TYPES) * len(RARITY_LEVELS)
