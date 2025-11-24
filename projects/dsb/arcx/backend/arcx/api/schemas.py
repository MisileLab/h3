"""API request/response schemas for overlay communication"""

from typing import Literal, Optional, List, Dict
from pydantic import BaseModel, Field


class EVResponse(BaseModel):
    """Expected Value response for current game state"""

    ev_stay: float = Field(description="Expected value for continuing to farm")
    ev_extract: float = Field(description="Expected value for extracting now")
    delta_ev: float = Field(description="EV_stay - EV_extract")

    recommendation: Literal["stay", "extract", "neutral"] = Field(
        description="Action recommendation based on delta"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")

    # UI display
    message: str = Field(description="Human-readable recommendation message")
    color: Literal["green", "yellow", "red"] = Field(description="Color indicator")

    # Metadata
    timestamp: float = Field(description="Unix timestamp")
    risk_profile: str = Field(description="Current risk profile setting")


class ConfigUpdate(BaseModel):
    """Configuration update from overlay UI"""

    risk_profile: Literal["safe", "neutral", "aggressive"] = Field(
        description="User's risk tolerance"
    )


class FeedbackSubmission(BaseModel):
    """User feedback on a recommendation"""

    run_id: str = Field(description="Current run/raid identifier")
    decision_idx: int = Field(description="Index of the decision point")
    timestamp: float = Field(description="Unix timestamp of feedback")
    rating: Literal["bad", "good"] = Field(description="User's rating")
    context: dict = Field(default_factory=dict, description="Additional context")


class SystemStatus(BaseModel):
    """System health and status"""

    is_capturing: bool = Field(description="Screen capture active")
    is_model_loaded: bool = Field(description="ML model loaded")
    buffer_frames: int = Field(description="Frames in ring buffer")
    device_backend: str = Field(description="ML device backend (cuda/rocm/dml/cpu)")
    fps: float = Field(description="Current capture FPS")
    inference_time_ms: float = Field(description="Last inference latency")


class StartRunRequest(BaseModel):
    """Start a new run/raid"""

    map_id: str = Field(default="unknown", description="Map identifier")
    metadata: dict = Field(default_factory=dict, description="Additional run metadata")


class DetectedItemSchema(BaseModel):
    """Individual detected item from YOLO"""

    item_type: str = Field(description="Type of item (e.g., weapon, armor)")
    rarity: str = Field(description="Rarity level (e.g., epic, rare)")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    estimated_value: float = Field(description="Estimated value of this item")
    bbox: List[float] = Field(description="Bounding box [x1, y1, x2, y2]")


class AutoValuationResult(BaseModel):
    """Automatic valuation result from YOLO detection"""

    total_value: float = Field(description="Total calculated loot value")
    items: List[DetectedItemSchema] = Field(description="List of detected items")
    num_items: int = Field(description="Total number of items detected")
    avg_confidence: float = Field(ge=0.0, le=1.0, description="Average confidence")
    value_breakdown: Dict[str, float] = Field(
        description="Value breakdown by item type"
    )
    rarity_counts: Dict[str, int] = Field(description="Count of items by rarity")
    phase_multiplier: float = Field(description="Game phase value multiplier applied")


class ValuateScreenshotRequest(BaseModel):
    """Request to valuate a screenshot"""

    screenshot_base64: str = Field(description="Base64-encoded screenshot image")
    game_phase: Optional[str] = Field(
        default="mid_wipe",
        description="Game phase for value adjustment"
    )


class EndRunRequest(BaseModel):
    """End current run with results"""

    run_id: str = Field(description="Run identifier")
    auto_valuation: AutoValuationResult = Field(
        description="Automatic valuation result from YOLO"
    )
    total_time_sec: float = Field(description="Total raid duration")
    success: bool = Field(description="Whether extraction was successful")
    action_taken: Literal["stay", "extract"] = Field(
        description="Final action taken (for last decision)"
    )
