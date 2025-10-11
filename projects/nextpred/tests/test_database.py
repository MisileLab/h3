"""Tests for database module."""

import pytest
import tempfile
from pathlib import Path

from database import DatabaseManager


@pytest.fixture
def temp_db():
  """Create a temporary database for testing."""
  with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    db_path = f.name

  db = DatabaseManager(db_path)
  yield db
  db.close()
  Path(db_path).unlink(missing_ok=True)


def test_database_initialization(temp_db):
  """Test database initialization."""
  assert temp_db.db_path.exists()

  # Check that tables were created
  cursor = temp_db.conn.cursor()
  tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

  table_names = [row[0] for row in tables]
  expected_tables = ["events", "patterns", "feedbacks", "settings"]

  for table in expected_tables:
    assert table in table_names


def test_log_event(temp_db):
  """Test event logging."""
  event_id = temp_db.log_event("test_app.exe", 1234, "test_session")
  assert event_id > 0

  # Verify event was logged
  cursor = temp_db.conn.cursor()
  row = cursor.execute(
    "SELECT app_name, process_id, session_id FROM events WHERE id = ?", (event_id,)
  ).fetchone()

  assert row is not None
  assert row[0] == "test_app.exe"
  assert row[1] == 1234
  assert row[2] == "test_session"


def test_get_recent_events(temp_db):
  """Test getting recent events."""
  # Log some test events
  temp_db.log_event("app1.exe", 1001, "session1")
  temp_db.log_event("app2.exe", 1002, "session1")
  temp_db.log_event("app3.exe", 1003, "session2")

  events = temp_db.get_recent_events(limit=2)
  assert len(events) == 2

  # Should be in reverse chronological order
  assert events[0]["app_name"] == "app3.exe"
  assert events[1]["app_name"] == "app2.exe"


def test_upsert_pattern_new(temp_db):
  """Test creating a new pattern."""
  sequence = ["app1.exe", "app2.exe", "app3.exe"]
  pattern_id = temp_db.upsert_pattern(sequence, 3, 3)

  assert pattern_id > 0

  # Verify pattern was created
  patterns = temp_db.get_active_patterns()
  assert len(patterns) == 1
  assert patterns[0]["sequence"] == sequence
  assert patterns[0]["occurrence_count"] == 3
  assert patterns[0]["confidence"] == 100.0


def test_upsert_pattern_update(temp_db):
  """Test updating an existing pattern."""
  sequence = ["app1.exe", "app2.exe"]

  # Create initial pattern
  pattern_id1 = temp_db.upsert_pattern(sequence, 2, 2)

  # Update pattern
  pattern_id2 = temp_db.upsert_pattern(sequence, 1, 1)

  # Should be the same pattern
  assert pattern_id1 == pattern_id2

  patterns = temp_db.get_active_patterns()
  assert len(patterns) == 1
  assert patterns[0]["occurrence_count"] == 3  # 2 + 1
  assert patterns[0]["total_attempts"] == 3  # 2 + 1


def test_add_feedback(temp_db):
  """Test adding feedback."""
  # Create a pattern first
  sequence = ["app1.exe", "app2.exe"]
  pattern_id = temp_db.upsert_pattern(sequence, 2, 2)

  # Add feedback
  feedback_id = temp_db.add_feedback(pattern_id, "accept")
  assert feedback_id > 0

  # Verify feedback was added
  cursor = temp_db.conn.cursor()
  row = cursor.execute(
    "SELECT pattern_id, action FROM feedbacks WHERE id = ?", (feedback_id,)
  ).fetchone()

  assert row is not None
  assert row[0] == pattern_id
  assert row[1] == "accept"


def test_delete_pattern_via_feedback(temp_db):
  """Test deleting pattern via feedback."""
  # Create a pattern
  sequence = ["app1.exe", "app2.exe"]
  pattern_id = temp_db.upsert_pattern(sequence, 2, 2)

  # Delete via feedback
  temp_db.add_feedback(pattern_id, "delete")

  # Pattern should be inactive
  patterns = temp_db.get_active_patterns()
  assert len(patterns) == 0


def test_get_pattern_stats(temp_db):
  """Test getting pattern statistics."""
  # Add some test data
  temp_db.log_event("app1.exe", 1001, "session1")
  temp_db.log_event("app2.exe", 1002, "session1")
  temp_db.log_event("app1.exe", 1003, "session2")

  temp_db.upsert_pattern(["app1.exe", "app2.exe"], 2, 2)
  temp_db.upsert_pattern(["app2.exe", "app1.exe"], 1, 1)

  stats = temp_db.get_pattern_stats()

  assert stats["total_patterns"] == 2
  assert stats["total_events"] == 3
  assert stats["average_confidence"] == 75.0  # (100 + 50) / 2
  assert len(stats["top_apps"]) == 2
  assert stats["top_apps"][0][0] == "app1.exe"  # Most common


if __name__ == "__main__":
  pytest.main([__file__])
