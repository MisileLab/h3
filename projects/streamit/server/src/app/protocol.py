"""
Protocol message definitions and validation using Pydantic.
"""

from pydantic import BaseModel
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

    type = "start"
    lang: str = "auto"
    clientSessionId: str
    platformHint: str = "unknown"


class StopMessage(BaseModel):
    """Client stop message."""

    type = "stop"


class PartialTranscriptMessage(BaseModel):
    """Server partial transcript message."""

    type = "partial"
    text: str


class FinalTranscriptMessage(BaseModel):
    """Server final transcript message."""

    type = "final"
    text: str


class InfoMessage(BaseModel):
    """Server info message (latency, usage)."""

    type = "info"
    latencyMs: int | None = None
    secondsUsed: float | None = None
    message: str | None = None


class ErrorMessage(BaseModel):
    """Server error message."""

    type = "error"
    message: str
    code: str | None = None


InboundMessage = StartMessage | StopMessage

OutboundMessage = (
    PartialTranscriptMessage | FinalTranscriptMessage | InfoMessage | ErrorMessage
)
