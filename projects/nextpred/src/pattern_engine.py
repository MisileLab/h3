"""Pattern recognition engine for Next Action Predictor."""

import logging
from collections import defaultdict
from datetime import datetime, time
from typing import Optional, Any, List

from database import DatabaseManager


class PatternEngine:
  """Engine for detecting and analyzing app usage patterns."""

  def __init__(
    self, db_manager: DatabaseManager, min_occurrences: int = 3, min_confidence: float = 60.0
  ) -> None:
    """Initialize pattern engine.

    Args:
      db_manager: Database manager instance
      min_occurrences: Minimum occurrences for pattern recognition
      min_confidence: Minimum confidence threshold for suggestions
    """
    self.db: DatabaseManager = db_manager
    self.min_occurrences: int = min_occurrences
    self.min_confidence: float = min_confidence

    logging.info(
      f"Pattern engine initialized (min_occurrences: {min_occurrences}, min_confidence: {min_confidence}%)"
    )

  def analyze_sessions(self, days: int = 7) -> None:
    """Analyze recent sessions and extract patterns.

    Args:
      days: Number of days to look back for analysis
    """
    logging.info(f"Analyzing sessions from last {days} days")

    sessions: List[List[dict[str, Any]]] = self.db.get_recent_sessions(days=days)
    sequences: dict[tuple[str, ...], int] = defaultdict(int)
    sequence_attempts: dict[tuple[str, ...], int] = defaultdict(int)

    for session in sessions:
      if len(session) < 2:  # Need at least 2 apps for a pattern
        continue

      apps: List[str] = [event["app_name"] for event in session]

      # Generate sequences of 2-5 apps
      for length in range(2, min(6, len(apps) + 1)):
        for i in range(len(apps) - length + 1):
          seq: tuple[str, ...] = tuple(apps[i : i + length])
          sequences[seq] += 1

          # Count attempts (when first app of sequence appears)
          if i == 0:
            sequence_attempts[seq] += 1

    # Process sequences and save patterns
    patterns_created: int = 0
    patterns_updated: int = 0

    for seq, count in sequences.items():
      if count >= self.min_occurrences:
        attempts: int = sequence_attempts.get(seq, count)
        confidence: float = (count / attempts) * 100 if attempts > 0 else 0.0

        # Add context information
        context: dict[str, Any] = self._extract_context(sessions, seq)

        # Check if pattern already exists
        existing_patterns: List[dict[str, Any]] = self.db.get_active_patterns()
        sequence_str: List[str] = list(seq)

        existing_found: bool = False
        for pattern in existing_patterns:
          if pattern["sequence"] == sequence_str:
            # Update existing pattern
            self.db.upsert_pattern(sequence_str, count, attempts, context)
            patterns_updated += 1
            existing_found = True
            break

        if not existing_found:
          # Create new pattern
          self.db.upsert_pattern(sequence_str, count, attempts, context)
          patterns_created += 1

    logging.info(f"Pattern analysis complete: {patterns_created} new, {patterns_updated} updated")

  def _extract_context(
    self, sessions: List[List[dict[str, Any]]], sequence: tuple[str, ...]
  ) -> dict[str, Any]:
    """Extract context information for a sequence.

    Args:
      sessions: List of sessions to analyze
      sequence: Sequence to extract context for

    Returns:
      Context dictionary
    """
    times_of_day: List[int] = []  # Hour of day (0-23)
    weekdays: List[int] = []  # Day of week (0=Monday, 6=Sunday)

    for session in sessions:
      apps: List[str] = [event["app_name"] for event in session]

      # Check if this session contains our sequence
      for i in range(len(apps) - len(sequence) + 1):
        if tuple(apps[i : i + len(sequence)]) == sequence:
          # Get the time of the first event in sequence
          if i < len(session):
            event_time: datetime = session[i]["started_at"]
            times_of_day.append(event_time.hour)
            weekdays.append(event_time.weekday())
          break

    context: dict[str, Any] = {}

    if times_of_day:
      avg_hour: float = sum(times_of_day) / len(times_of_day)
      if 6 <= avg_hour < 12:
        context["time_of_day"] = "morning"
      elif 12 <= avg_hour < 18:
        context["time_of_day"] = "afternoon"
      elif 18 <= avg_hour < 22:
        context["time_of_day"] = "evening"
      else:
        context["time_of_day"] = "night"

      context["avg_hour"] = round(avg_hour, 1)

    if weekdays:
      weekday_count: int = sum(1 for day in weekdays if day < 5)  # Monday-Friday
      context["weekday_preference"] = weekday_count / len(weekdays) >= 0.7

    return context

  def check_pattern_match(self, recent_apps: List[str]) -> Optional[dict[str, Any]]:
    """Check if current executed apps match any pattern.

    Args:
      recent_apps: List of recently executed apps

    Returns:
      Suggestion dictionary or None if no match
    """
    if not recent_apps:
      return None

    patterns: List[dict[str, Any]] = self.db.get_active_patterns()

    for pattern in patterns:
      sequence: List[str] = pattern["sequence"]

      # Check if n-1 elements of pattern match recent apps
      if len(recent_apps) >= len(sequence) - 1:
        recent_subsequence: List[str] = recent_apps[-(len(sequence) - 1) :]
        if recent_subsequence == sequence[:-1]:
          if pattern["confidence"] >= self.min_confidence:
            return {
              "pattern_id": pattern["id"],
              "next_app": sequence[-1],
              "confidence": pattern["confidence"],
              "sequence": sequence,
              "context": pattern.get("context", {}),
            }

    return None

  def get_pattern_suggestions(self, limit: int = 5) -> List[dict[str, Any]]:
    """Get top pattern suggestions based on confidence.

    Args:
      limit: Maximum number of suggestions to return

    Returns:
      List of pattern suggestions
    """
    patterns: List[dict[str, Any]] = self.db.get_active_patterns()

    # Filter by minimum confidence and sort
    valid_patterns: List[dict[str, Any]] = [
      pattern for pattern in patterns if pattern["confidence"] >= self.min_confidence
    ]

    # Sort by confidence and recency
    valid_patterns.sort(key=lambda p: (p["confidence"], p["last_seen"]), reverse=True)

    return valid_patterns[:limit]

  def update_pattern_feedback(self, pattern_id: int, action: str) -> None:
    """Update pattern based on user feedback.

    Args:
      pattern_id: Pattern ID
      action: User action ('accept', 'reject', 'delete')
    """
    self.db.add_feedback(pattern_id, action)

    if action == "accept":
      logging.info(f"User accepted suggestion for pattern {pattern_id}")
    elif action == "reject":
      logging.info(f"User rejected suggestion for pattern {pattern_id}")
    elif action == "delete":
      logging.info(f"User deleted pattern {pattern_id}")

  def get_pattern_statistics(self) -> dict[str, Any]:
    """Get comprehensive pattern statistics.

    Returns:
      Statistics dictionary
    """
    patterns: List[dict[str, Any]] = self.db.get_active_patterns()

    if not patterns:
      return {
        "total_patterns": 0,
        "average_confidence": 0.0,
        "confidence_distribution": {},
        "sequence_length_distribution": {},
        "most_common_apps": [],
        "time_preferences": {},
      }

    # Basic stats
    total_patterns: int = len(patterns)
    avg_confidence: float = sum(p["confidence"] for p in patterns) / total_patterns

    # Confidence distribution
    confidence_ranges: dict[str, int] = {
      "90-100%": 0,
      "80-89%": 0,
      "70-79%": 0,
      "60-69%": 0,
      "50-59%": 0,
      "below 50%": 0,
    }

    for pattern in patterns:
      conf: float = pattern["confidence"]
      if conf >= 90:
        confidence_ranges["90-100%"] += 1
      elif conf >= 80:
        confidence_ranges["80-89%"] += 1
      elif conf >= 70:
        confidence_ranges["70-79%"] += 1
      elif conf >= 60:
        confidence_ranges["60-69%"] += 1
      elif conf >= 50:
        confidence_ranges["50-59%"] += 1
      else:
        confidence_ranges["below 50%"] += 1

    # Sequence length distribution
    length_dist: dict[int, int] = defaultdict(int)
    for pattern in patterns:
      length_dist[len(pattern["sequence"])] += 1

    # Most common apps in patterns
    app_counts: dict[str, int] = defaultdict(int)
    for pattern in patterns:
      for app in pattern["sequence"]:
        app_counts[app] += 1

    top_apps: List[tuple[str, int]] = sorted(app_counts.items(), key=lambda x: x[1], reverse=True)[
      :10
    ]

    # Time preferences
    time_prefs: dict[str, int] = defaultdict(int)
    for pattern in patterns:
      context: dict[str, Any] = pattern.get("context", {})
      if "time_of_day" in context:
        time_prefs[context["time_of_day"]] += 1

    return {
      "total_patterns": total_patterns,
      "average_confidence": round(avg_confidence, 1),
      "confidence_distribution": confidence_ranges,
      "sequence_length_distribution": dict(length_dist),
      "most_common_apps": top_apps,
      "time_preferences": dict(time_prefs),
    }

  def find_similar_patterns(
    self, sequence: List[str], min_overlap: int = 2
  ) -> List[dict[str, Any]]:
    """Find patterns that overlap with given sequence.

    Args:
      sequence: Sequence to find similar patterns for
      min_overlap: Minimum number of apps that must overlap

    Returns:
      List of similar patterns
    """
    patterns: List[dict[str, Any]] = self.db.get_active_patterns()
    similar_patterns: List[dict[str, Any]] = []

    for pattern in patterns:
      pattern_seq: List[str] = pattern["sequence"]

      # Calculate overlap
      overlap: int = len(set(sequence) & set(pattern_seq))

      if overlap >= min_overlap:
        similarity: float = overlap / max(len(sequence), len(pattern_seq))
        similar_patterns.append(
          {**pattern, "overlap_count": overlap, "similarity": round(similarity, 2)}
        )

    # Sort by similarity and confidence
    similar_patterns.sort(key=lambda p: (p["similarity"], p["confidence"]), reverse=True)

    return similar_patterns
