"""Type definitions for Next Action Predictor."""

from datetime import datetime
from typing import TypedDict, Optional, Any


class EventDict(TypedDict):
  """Type for app execution event dictionary."""

  id: int
  app_name: str
  process_id: int
  started_at: datetime
  ended_at: Optional[datetime]
  session_id: str


class PatternDict(TypedDict):
  """Type for pattern dictionary."""

  id: int
  sequence: list[str]
  occurrence_count: int
  total_attempts: int
  confidence: float
  last_seen: datetime
  created_at: datetime
  is_active: bool
  context: Optional[dict[str, Any]]


class FeedbackDict(TypedDict):
  """Type for user feedback dictionary."""

  id: int
  pattern_id: int
  action: str
  timestamp: datetime


class SuggestionDict(TypedDict):
  """Type for suggestion dictionary."""

  pattern_id: int
  next_app: str
  confidence: float


class SessionDict(TypedDict):
  """Type for session dictionary."""

  session_id: str
  events: list[EventDict]
  start_time: datetime
  end_time: Optional[datetime]


class ConfigDict(TypedDict):
  """Type for configuration dictionary."""

  min_occurrences: int
  min_confidence: float
  session_timeout_minutes: int
  notification_cooldown_hours: int
  excluded_processes: set[str]
