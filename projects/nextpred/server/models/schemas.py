"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    TAB_SWITCH = "tab_switch"
    NAVIGATION = "navigation"
    PAGE_LOAD = "page_load"
    SCROLL = "scroll"
    CLICK = "click"
    SEARCH = "search"


class PredictionType(str, Enum):
    TAB_SWITCH = "tab_switch"
    SEARCH = "search"
    SCROLL = "scroll"


class EventData(BaseModel):
    """Base event data structure"""
    tabId: Optional[int] = None
    windowId: Optional[int] = None
    url: Optional[str] = None
    title: Optional[str] = None
    index: Optional[int] = None
    active: Optional[bool] = None
    urlDomain: Optional[str] = None
    scrollPosition: Optional[float] = None
    pageHeight: Optional[int] = None
    scrollTop: Optional[int] = None
    targetUrl: Optional[str] = None
    linkText: Optional[str] = None
    elementTag: Optional[str] = None
    elementClass: Optional[str] = None
    position: Optional[Dict[str, float]] = None
    query: Optional[str] = None
    searchUrl: Optional[str] = None
    source: Optional[str] = None
    isGoogleSearch: Optional[bool] = None
    isInternalSearch: Optional[bool] = None
    timeOnPage: Optional[float] = None
    viewportHeight: Optional[int] = None


class Event(BaseModel):
    """Individual event structure"""
    id: Optional[str] = None
    type: EventType
    timestamp: int
    data: EventData

    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v <= 0:
            raise ValueError('Timestamp must be positive')
        return v


class EventBatch(BaseModel):
    """Batch of events from extension"""
    events: List[Event]
    metadata: Optional[Dict[str, Any]] = None

    @validator('events')
    def validate_events_not_empty(cls, v):
        if not v:
            raise ValueError('Events list cannot be empty')
        return v


class EventResponse(BaseModel):
    """Response for event upload"""
    success: bool
    stored: int
    timestamp: datetime
    error: Optional[str] = None


class ModelVersionResponse(BaseModel):
    """Model version information"""
    version: str
    timestamp: datetime
    accuracy_metrics: Optional[Dict[str, float]] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None


class FeedbackType(str, Enum):
    PREDICTION_SELECTED = "prediction_selected"
    PREDICTION_REJECTED = "prediction_rejected"
    GENERAL = "general"
    BUG_REPORT = "bug_report"


class FeedbackData(BaseModel):
    """Feedback data structure"""
    predictionId: Optional[str] = None
    selectedIndex: Optional[int] = None
    context: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)


class FeedbackRequest(BaseModel):
    """Feedback request from extension"""
    feedback_type: FeedbackType
    feedback: FeedbackData
    metadata: Optional[Dict[str, Any]] = None


class FeedbackResponse(BaseModel):
    """Response for feedback submission"""
    success: bool
    feedback_id: str
    timestamp: datetime
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    healthy: bool
    server: str
    version: str
    timestamp: datetime
    components: Optional[Dict[str, str]] = None
    error: Optional[str] = None


class UserStatsResponse(BaseModel):
    """User statistics response"""
    user_id: Optional[str] = None
    total_events: int
    events_by_type: Dict[str, int]
    unique_domains: int
    top_domains: List[Dict[str, Any]]
    avg_session_duration: float
    predictions_accuracy: Optional[Dict[str, float]] = None
    period_days: int
    timestamp: datetime


class PredictionData(BaseModel):
    """Individual prediction data"""
    type: PredictionType
    url: Optional[str] = None
    title: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str  # "model" or "rule_based"
    tabId: Optional[int] = None
    domain: Optional[str] = None
    query: Optional[str] = None
    scrollPosition: Optional[float] = None


class PredictionRequest(BaseModel):
    """Prediction request from extension"""
    context: Dict[str, Any]
    current_tab: Dict[str, Any]
    user_id: Optional[str] = None


class PredictionResponse(BaseModel):
    """Prediction response to extension"""
    predictions: List[PredictionData]
    timestamp: datetime
    model_version: Optional[str] = None
    processing_time_ms: Optional[float] = None


class TrainingConfig(BaseModel):
    """Training configuration"""
    epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=64, ge=1, le=512)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)
    early_stopping: bool = True
    patience: int = Field(default=5, ge=1, le=20)


class TrainingJob(BaseModel):
    """Training job information"""
    id: str
    status: str  # "pending", "running", "completed", "failed"
    config: TrainingConfig
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    top1_accuracy: float
    top3_accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    model_size_mb: Optional[float] = None


class SystemStatus(BaseModel):
    """System status information"""
    server: Dict[str, Any]
    database: Dict[str, Any]
    qdrant: Dict[str, Any]
    model: Dict[str, Any]
    metrics: Dict[str, Any]