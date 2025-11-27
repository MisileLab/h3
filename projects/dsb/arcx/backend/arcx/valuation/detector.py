"""GPT-5-nano vision model based item detector using pydantic_ai."""

import base64
import logging
import os
from io import BytesIO
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from .config import (
    VISION_MODEL_NAME,
    VISION_API_TIMEOUT,
    MIN_DETECTION_CONFIDENCE,
    ITEM_TYPES,
    RARITY_LEVELS,
)

logger = logging.getLogger(__name__)

# Type aliases for better type hints
ItemType = Literal["weapon", "armor", "material", "consumable", "mod", "currency"]
RarityLevel = Literal["legendary", "epic", "rare", "uncommon", "common"]


class DetectedItemSchema(BaseModel):
    """Schema for a single detected item with structured output."""

    item_type: ItemType = Field(..., description="Type of the detected item")
    rarity: RarityLevel = Field(..., description="Rarity level of the item")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {v}")
        return v


class ItemDetectionResult(BaseModel):
    """Result from item detection containing list of detected items."""

    items: List[DetectedItemSchema] = Field(
        default_factory=list, description="List of detected items"
    )


# System prompt for the vision agent
SYSTEM_PROMPT = """You are an expert item analyzer for extraction shooter games.
Analyze screenshots from extraction screens and identify all visible items with their types and rarities.

Be precise and confident in your classifications. Count each distinct item separately.
If you see multiple items of the same type and rarity, list them as separate entries."""


class ItemDetector:
    """
    GPT-5-nano vision model based item detector using pydantic_ai.

    This detector uses pydantic_ai with structured output to reliably identify
    items and their rarities from game screenshots.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = VISION_MODEL_NAME,
        confidence: float = MIN_DETECTION_CONFIDENCE,
    ):
        """
        Initialize item detector.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Vision model name (default: gpt-5-nano)
            confidence: Minimum confidence threshold for detections
        """
        self.confidence = confidence
        self.model_name = model

        # Get API key
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Set it in .env file or pass api_key parameter."
                )

        # Initialize OpenAI model
        self.openai_model = OpenAIModel(
            model_name=model,
            api_key=api_key,
        )

        # Create pydantic_ai agent with structured output
        self.agent = Agent(
            model=self.openai_model,
            result_type=ItemDetectionResult,
            system_prompt=SYSTEM_PROMPT,
        )

        logger.info(f"ItemDetector initialized with model: {model}")

    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode numpy image to base64 PNG.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            Base64 encoded PNG image string
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Encode to PNG in memory
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Base64 encode
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        logger.debug(f"Encoded image size: {len(encoded)} bytes")

        return encoded

    def detect(self, image: np.ndarray) -> List[Tuple[str, str, float]]:
        """
        Detect items in image using vision model with structured output.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            List of detections, each containing:
                - item_type: Type of item
                - rarity: Rarity level
                - confidence: Detection confidence score
        """
        try:
            # Encode image
            base64_image = self._encode_image(image)

            # Create data URL for the image
            image_url = f"data:image/png;base64,{base64_image}"

            # User prompt with image
            user_prompt = f"""Analyze this extraction screen screenshot and identify all items.

For each item you detect, provide:
- item_type: one of [weapon, armor, material, consumable, mod, currency]
- rarity: one of [legendary, epic, rare, uncommon, common]
- confidence: your confidence score (0.0-1.0)

Count each distinct item separately. If there are multiple items of the same type and rarity, list them as separate entries.

Image: {image_url}"""

            logger.debug("Making API call with pydantic_ai agent...")

            # Run agent with structured output
            result = self.agent.run_sync(user_prompt)

            # Extract structured data
            detection_result: ItemDetectionResult = result.data

            logger.info(f"Received {len(detection_result.items)} items from API")

            # Filter by confidence threshold and convert to tuple format
            detections = []
            for item in detection_result.items:
                if item.confidence >= self.confidence:
                    detections.append((item.item_type, item.rarity, item.confidence))
                else:
                    logger.debug(
                        f"Filtered out {item.item_type} "
                        f"({item.rarity}) with confidence {item.confidence:.2f}"
                    )

            logger.info(
                f"Detected {len(detections)} items "
                f"(filtered from {len(detection_result.items)})"
            )
            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            return []
