"""
Protocol message definitions and validation using Pydantic.
"""

from pydantic import BaseModel
from typing import Optional, Literal
from enum import Enum


class MessageType(str, Enum):
    """Message types for client-server communication."""

    START = "start"
    STOP = "stop"
    PARTIAL = "partial"
    FINAL = "final"
    INFO = "info"
    ERROR = "error"


class StartMessage(BaseModel):
    """Client start message."""

    type: Literal["start"] = "start"
    lang: str = "auto"
    clientSessionId: str
    platformHint: str = "unknown"


class StopMessage(BaseModel):
    """Client stop message."""

    type: Literal["stop"] = "stop"


class PartialTranscriptMessage(BaseModel):
    """Server partial transcript message."""

    type: Literal["partial"] = "partial"
    text: str


class FinalTranscriptMessage(BaseModel):
    """Server final transcript message."""

    type: Literal["final"] = "final"
    text: str


class InfoMessage(BaseModel):
    """Server info message (latency, usage)."""

    type: Literal["info"] = "info"
    latencyMs: Optional[int] = None
    secondsUsed: Optional[float] = None
    message: Optional[str] = None


class ErrorMessage(BaseModel):
    """Server error message."""

    type: Literal["error"] = "error"
    message: str
    code: Optional[str] = None


InboundMessage = StartMessage | StopMessage

OutboundMessage = (
    PartialTranscriptMessage | FinalTranscriptMessage | InfoMessage | ErrorMessage
)
