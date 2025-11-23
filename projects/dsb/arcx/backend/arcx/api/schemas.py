"""API request/response schemas for overlay communication"""

from typing import Literal
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


class EndRunRequest(BaseModel):
    """End current run with results"""

    run_id: str = Field(description="Run identifier")
    final_loot_value: float = Field(description="Total loot value obtained")
    total_time_sec: float = Field(description="Total raid duration")
    success: bool = Field(description="Whether extraction was successful")
    action_taken: Literal["stay", "extract"] = Field(
        description="Final action taken (for last decision)"
    )
