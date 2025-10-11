"""
Pydantic schemas for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class SafetyDecision(str, Enum):
    """Safety decision enum."""
    SAFE = "safe"
    REVIEW = "review"
    BLOCK = "block"


class PredictionRequest(BaseModel):
    """Request schema for prediction."""
    text: str = Field(..., description="Text to analyze for jailbreak attempts")
    threshold: Optional[float] = Field(0.5, description="Custom threshold for classification")
    return_probabilities: Optional[bool] = Field(True, description="Return probability scores")


class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    text: str = Field(..., description="Original text")
    predicted_class: int = Field(..., description="Predicted class (0=safe, 1=unsafe)")
    decision: SafetyDecision = Field(..., description="Safety decision")
    confidence: float = Field(..., description="Confidence score")
    unsafe_probability: float = Field(..., description="Probability of being unsafe")
    safe_probability: float = Field(..., description="Probability of being safe")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Full probability distribution")
    processing_time: float = Field(..., description="Processing time in seconds")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""
    texts: List[str] = Field(..., description="List of texts to analyze")
    threshold: Optional[float] = Field(0.5, description="Custom threshold for classification")
    return_probabilities: Optional[bool] = Field(True, description="Return probability scores")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""
    results: List[PredictionResponse] = Field(..., description="Prediction results")
    total_processed: int = Field(..., description="Total number of texts processed")
    processing_time: float = Field(..., description="Total processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type (full/lora)")
    num_parameters: int = Field(..., description="Total number of parameters")
    max_length: int = Field(..., description="Maximum sequence length")
    version: str = Field(..., description="Model version")