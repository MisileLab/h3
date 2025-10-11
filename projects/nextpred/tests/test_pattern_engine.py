"""Tests for pattern engine module."""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from database import DatabaseManager
from pattern_engine import PatternEngine


@pytest.fixture
def temp_db():
  """Create a temporary database for testing."""
  with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    db_path = f.name

  db = DatabaseManager(db_path)
  yield db
  db.close()
  Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def pattern_engine(temp_db):
  """Create a pattern engine for testing."""
  return PatternEngine(temp_db, min_occurrences=2, min_confidence=50.0)


def test_pattern_engine_initialization(pattern_engine):
  """Test pattern engine initialization."""
  assert pattern_engine.min_occurrences == 2
  assert pattern_engine.min_confidence == 50.0
  assert pattern_engine.db is not None


def test_pattern_detection_simple(pattern_engine, temp_db):
  """Test simple pattern detection."""
  # Create test sessions with repeated pattern
  sessions = [
    [
      {"app_name": "chrome.exe", "started_at": datetime.now()},
      {"app_name": "notion.exe", "started_at": datetime.now()},
      {"app_name": "code.exe", "started_at": datetime.now()},
    ],
    [
      {"app_name": "chrome.exe", "started_at": datetime.now()},
      {"app_name": "notion.exe", "started_at": datetime.now()},
      {"app_name": "code.exe", "started_at": datetime.now()},
    ],
  ]

  # Manually insert session data
  for i, session in enumerate(sessions):
    session_id = f"test_session_{i}"
    for event in session:
      temp_db.log_event(event["app_name"], 1000 + i, session_id)

  # Analyze patterns
  pattern_engine.analyze_sessions(days=1)

  # Check that patterns were detected
  patterns = temp_db.get_active_patterns()
  assert len(patterns) > 0

  # Should find the chrome -> notion -> code pattern
  chrome_notion_code = [
    p for p in patterns if p["sequence"] == ["chrome.exe", "notion.exe", "code.exe"]
  ]
  assert len(chrome_notion_code) == 1
  assert chrome_notion_code[0]["occurrence_count"] == 2


def test_pattern_match_detection(pattern_engine, temp_db):
  """Test pattern matching for suggestions."""
  # Create a pattern
  sequence = ["app1.exe", "app2.exe", "app3.exe"]
  temp_db.upsert_pattern(sequence, 3, 3)  # 100% confidence

  # Test with matching recent apps
  recent_apps = ["app1.exe", "app2.exe"]
  suggestion = pattern_engine.check_pattern_match(recent_apps)

  assert suggestion is not None
  assert suggestion["next_app"] == "app3.exe"
  assert suggestion["confidence"] == 100.0
  assert suggestion["pattern_id"] > 0


def test_pattern_match_insufficient_apps(pattern_engine, temp_db):
  """Test pattern matching with insufficient recent apps."""
  # Create a pattern
  sequence = ["app1.exe", "app2.exe", "app3.exe"]
  temp_db.upsert_pattern(sequence, 3, 3)

  # Test with insufficient recent apps
  recent_apps = ["app1.exe"]
  suggestion = pattern_engine.check_pattern_match(recent_apps)

  assert suggestion is None


def test_pattern_match_low_confidence(pattern_engine, temp_db):
  """Test pattern matching with low confidence."""
  # Create a pattern with low confidence
  sequence = ["app1.exe", "app2.exe", "app3.exe"]
  temp_db.upsert_pattern(sequence, 1, 3)  # 33% confidence

  # Test with matching recent apps
  recent_apps = ["app1.exe", "app2.exe"]
  suggestion = pattern_engine.check_pattern_match(recent_apps)

  # Should not suggest due to low confidence
  assert suggestion is None


def test_get_pattern_suggestions(pattern_engine, temp_db):
  """Test getting pattern suggestions."""
  # Create multiple patterns with different confidences
  temp_db.upsert_pattern(["app1.exe", "app2.exe"], 5, 5)  # 100%
  temp_db.upsert_pattern(["app3.exe", "app4.exe"], 3, 4)  # 75%
  temp_db.upsert_pattern(["app5.exe", "app6.exe"], 1, 3)  # 33%

  suggestions = pattern_engine.get_pattern_suggestions(limit=5)

  # Should only return high-confidence patterns
  assert len(suggestions) == 2

  # Should be sorted by confidence
  assert suggestions[0]["confidence"] >= suggestions[1]["confidence"]


def test_update_pattern_feedback(pattern_engine, temp_db):
  """Test updating pattern feedback."""
  # Create a pattern
  sequence = ["app1.exe", "app2.exe"]
  pattern_id = temp_db.upsert_pattern(sequence, 2, 2)

  # Add feedback
  pattern_engine.update_pattern_feedback(pattern_id, "accept")

  # Verify feedback was recorded
  cursor = temp_db.conn.cursor()
  row = cursor.execute(
    "SELECT action FROM feedbacks WHERE pattern_id = ?", (pattern_id,)
  ).fetchone()

  assert row is not None
  assert row[0] == "accept"


def test_delete_pattern_via_feedback(pattern_engine, temp_db):
  """Test deleting pattern via feedback."""
  # Create a pattern
  sequence = ["app1.exe", "app2.exe"]
  pattern_id = temp_db.upsert_pattern(sequence, 2, 2)

  # Delete via feedback
  pattern_engine.update_pattern_feedback(pattern_id, "delete")

  # Pattern should be inactive
  patterns = temp_db.get_active_patterns()
  assert len(patterns) == 0


def test_get_pattern_statistics(pattern_engine, temp_db):
  """Test getting pattern statistics."""
  # Create test patterns
  temp_db.upsert_pattern(["app1.exe", "app2.exe"], 5, 5)  # 100%
  temp_db.upsert_pattern(["app2.exe", "app3.exe"], 3, 4)  # 75%
  temp_db.upsert_pattern(["app1.exe", "app3.exe"], 2, 3)  # 67%

  stats = pattern_engine.get_pattern_statistics()

  assert stats["total_patterns"] == 3
  assert stats["average_confidence"] == 80.7  # (100 + 75 + 67) / 3
  assert "confidence_distribution" in stats
  assert "most_common_apps" in stats
  assert "sequence_length_distribution" in stats


def test_find_similar_patterns(pattern_engine, temp_db):
  """Test finding similar patterns."""
  # Create patterns
  temp_db.upsert_pattern(["app1.exe", "app2.exe", "app3.exe"], 3, 3)
  temp_db.upsert_pattern(["app1.exe", "app2.exe", "app4.exe"], 2, 2)
  temp_db.upsert_pattern(["app5.exe", "app6.exe"], 1, 1)

  # Find similar patterns
  similar = pattern_engine.find_similar_patterns(
    ["app1.exe", "app2.exe", "app7.exe"], min_overlap=2
  )

  # Should find 2 similar patterns (both share app1.exe and app2.exe)
  assert len(similar) == 2

  # Should be sorted by similarity
  assert all(s["similarity"] >= 0.5 for s in similar)


if __name__ == "__main__":
  pytest.main([__file__])
