"""Database management for Next Action Predictor."""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, List


class DatabaseManager:
  """SQLite database manager for app patterns and events."""

  def __init__(self, db_path: Optional[str] = None) -> None:
    """Initialize database manager.

    Args:
      db_path: Path to database file. If None, uses default location.
    """
    if db_path:
      self.db_path: Path = Path(db_path)
    else:
      # Use platform-specific data directory
      import platform

      if platform.system() == "Windows":
        data_dir = Path.home() / "AppData" / "Local" / "NextActionPredictor"
      else:
        data_dir = Path.home() / ".local" / "share" / "next-action-predictor"

      data_dir.mkdir(parents=True, exist_ok=True)
      self.db_path = data_dir / "app_patterns.db"

    self.db_path.parent.mkdir(parents=True, exist_ok=True)

    self.conn: sqlite3.Connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
    self.conn.row_factory = sqlite3.Row

    self._init_schema()
    logging.info(f"Database initialized at {self.db_path}")

  def _init_schema(self) -> None:
    """Initialize database schema with migration support."""
    cursor: sqlite3.Cursor = self.conn.cursor()
    
    # Get current database version
    cursor.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER DEFAULT 1)")
    cursor.execute("SELECT version FROM schema_version")
    result = cursor.fetchone()
    current_version = result['version'] if result else 1
    
    logging.info(f"Current database schema version: {current_version}")
    
    # Create initial tables if they don't exist
    self._create_initial_tables(cursor)
    
    # Run migrations based on current version
    if current_version < 2:
      self._migrate_to_v2(cursor)
    
    # Update schema version
    cursor.execute("UPDATE schema_version SET version = 2")
    
    self.conn.commit()
    logging.info("Database schema initialized and migrated")

  def _create_initial_tables(self, cursor: sqlite3.Cursor) -> None:
    """Create initial database tables."""
    # Events table - app execution events
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        app_name TEXT NOT NULL,
        process_id INTEGER,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ended_at TIMESTAMP,
        session_id TEXT NOT NULL
      )
    """)

    # Patterns table - learned app sequences
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sequence TEXT NOT NULL,
        occurrence_count INTEGER DEFAULT 1,
        total_attempts INTEGER DEFAULT 1,
        confidence REAL DEFAULT 0.0,
        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active BOOLEAN DEFAULT 1,
        context TEXT
      )
    """)

    # Feedback table - user responses to suggestions
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS feedbacks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_id INTEGER,
        action TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (pattern_id) REFERENCES patterns(id)
      )
    """)

    # Settings table - application settings
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
      )
    """)

    # Create indexes for better performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session_id ON events(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_started_at ON events(started_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_active ON patterns(is_active)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_last_seen ON patterns(last_seen)")

  def _migrate_to_v2(self, cursor: sqlite3.Cursor) -> None:
    """Migrate database schema to version 2."""
    logging.info("Migrating database to version 2...")
    
    try:
      # Check if new columns already exist
      cursor.execute("PRAGMA table_info(events)")
      columns = [column[1] for column in cursor.fetchall()]
      
      # Add new columns to events table if they don't exist
      if 'duration_seconds' not in columns:
        cursor.execute("ALTER TABLE events ADD COLUMN duration_seconds INTEGER DEFAULT 0")
        logging.info("Added duration_seconds column to events table")
      
      if 'mouse_x' not in columns:
        cursor.execute("ALTER TABLE events ADD COLUMN mouse_x INTEGER DEFAULT 0")
        logging.info("Added mouse_x column to events table")
      
      if 'mouse_y' not in columns:
        cursor.execute("ALTER TABLE events ADD COLUMN mouse_y INTEGER DEFAULT 0")
        logging.info("Added mouse_y column to events table")
      
      if 'window_title' not in columns:
        cursor.execute("ALTER TABLE events ADD COLUMN window_title TEXT DEFAULT ''")
        logging.info("Added window_title column to events table")
      
      # Create new indexes for performance
      cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_app_name ON events(app_name)")
      cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_duration ON events(duration_seconds)")
      
      logging.info("Database migration to version 2 completed successfully")
      
    except Exception as e:
      logging.error(f"Error during database migration: {e}")
      raise

  def log_event(self, app_name: str, process_id: int, session_id: str, 
                 duration_seconds: int = 0, mouse_x: int = 0, mouse_y: int = 0, 
                 window_title: str = '') -> int:
    """Log app execution event.

    Args:
      app_name: Name of the application
      process_id: Process ID
      session_id: Session identifier
      duration_seconds: How long the app was focused
      mouse_x: Mouse X position
      mouse_y: Mouse Y position
      window_title: Window title for context

    Returns:
      Event ID
    """
    cursor: sqlite3.Cursor = self.conn.cursor()
    
    # Check database schema version and use appropriate query
    cursor.execute("PRAGMA table_info(events)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'duration_seconds' in columns:
      # New schema with all columns
      cursor.execute(
        """
        INSERT INTO events (app_name, process_id, session_id, duration_seconds, 
                            mouse_x, mouse_y, window_title)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (app_name, process_id, session_id, duration_seconds, mouse_x, mouse_y, window_title),
      )
    else:
      # Old schema - use basic columns
      cursor.execute(
        """
        INSERT INTO events (app_name, process_id, session_id)
        VALUES (?, ?, ?)
        """,
        (app_name, process_id, session_id),
      )
    
    self.conn.commit()
    event_id: int = cursor.lastrowid or 0
    logging.debug(f"Logged event: {app_name} (PID: {process_id}, duration: {duration_seconds}s) in session {session_id}")
    return event_id

  def get_recent_sessions(self, days: int = 7) -> List[List[dict[str, Any]]]:
    """Get sessions from recent days.

    Args:
      days: Number of days to look back

    Returns:
      List of sessions, each containing list of events
    """
    cursor: sqlite3.Cursor = self.conn.cursor()
    cutoff_date: datetime = datetime.now() - timedelta(days=days)

    cursor.execute(
      """
      SELECT session_id, app_name, process_id, started_at, ended_at
      FROM events
      WHERE started_at >= ?
      ORDER BY session_id, started_at
      """,
      (cutoff_date.isoformat(),),
    )

    rows: List[sqlite3.Row] = cursor.fetchall()
    sessions: List[List[dict[str, Any]]] = []
    current_session: List[dict[str, Any]] = []
    current_session_id: Optional[str] = None

    for row in rows:
      session_id: str = row["session_id"]
      event: dict[str, Any] = {
        "app_name": row["app_name"],
        "process_id": row["process_id"],
        "started_at": datetime.fromisoformat(row["started_at"]),
        "ended_at": datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
      }

      if session_id != current_session_id:
        if current_session:
          sessions.append(current_session)
        current_session = [event]
        current_session_id = session_id
      else:
        current_session.append(event)

    if current_session:
      sessions.append(current_session)

    logging.debug(f"Retrieved {len(sessions)} sessions from last {days} days")
    return sessions

  def upsert_pattern(
    self,
    sequence: List[str],
    occurrence_count: int,
    total_attempts: Optional[int] = None,
    context: Optional[dict[str, Any]] = None,
  ) -> int:
    """Insert or update pattern.

    Args:
      sequence: List of app names in sequence
      occurrence_count: Number of times this pattern occurred
      total_attempts: Total attempts after first app
      context: Additional context data

    Returns:
      Pattern ID
    """
    cursor: sqlite3.Cursor = self.conn.cursor()
    sequence_json: str = json.dumps(sequence)
    context_json: Optional[str] = json.dumps(context) if context else None

    # Check if pattern already exists
    cursor.execute(
      "SELECT id, occurrence_count, total_attempts FROM patterns WHERE sequence = ? AND is_active = 1",
      (sequence_json,),
    )
    existing: Optional[sqlite3.Row] = cursor.fetchone()

    if existing:
      # Update existing pattern
      pattern_id: int = existing["id"]
      new_occurrence_count: int = existing["occurrence_count"] + occurrence_count
      new_total_attempts: int = total_attempts or existing["total_attempts"] + 1
      new_confidence: float = (new_occurrence_count / new_total_attempts) * 100

      cursor.execute(
        """
        UPDATE patterns 
        SET occurrence_count = ?, total_attempts = ?, confidence = ?, 
            last_seen = CURRENT_TIMESTAMP, context = ?
        WHERE id = ?
        """,
        (new_occurrence_count, new_total_attempts, new_confidence, context_json, pattern_id),
      )
      logging.debug(f"Updated pattern {pattern_id}: {sequence} (confidence: {new_confidence:.1f}%)")
    else:
      # Insert new pattern
      new_total_attempts = total_attempts or 1
      new_confidence: float = (occurrence_count / new_total_attempts) * 100

      cursor.execute(
        """
        INSERT INTO patterns (sequence, occurrence_count, total_attempts, confidence, context)
        VALUES (?, ?, ?, ?, ?)
        """,
        (sequence_json, occurrence_count, new_total_attempts, new_confidence, context_json),
      )
      pattern_id = cursor.lastrowid or 0
      logging.debug(
        f"Created new pattern {pattern_id}: {sequence} (confidence: {new_confidence:.1f}%)"
      )

    self.conn.commit()
    return pattern_id

  def get_active_patterns(self) -> List[dict[str, Any]]:
    """Get all active patterns.

    Returns:
      List of active patterns
    """
    cursor: sqlite3.Cursor = self.conn.cursor()
    cursor.execute(
      """
      SELECT id, sequence, occurrence_count, total_attempts, confidence, 
             last_seen, created_at, is_active, context
      FROM patterns
      WHERE is_active = 1
      ORDER BY confidence DESC, last_seen DESC
      """
    )

    patterns: List[dict[str, Any]] = []
    for row in cursor.fetchall():
      pattern: dict[str, Any] = {
        "id": row["id"],
        "sequence": json.loads(row["sequence"]),
        "occurrence_count": row["occurrence_count"],
        "total_attempts": row["total_attempts"],
        "confidence": row["confidence"],
        "last_seen": datetime.fromisoformat(row["last_seen"]),
        "created_at": datetime.fromisoformat(row["created_at"]),
        "is_active": bool(row["is_active"]),
        "context": json.loads(row["context"]) if row["context"] else None,
      }
      patterns.append(pattern)

    logging.debug(f"Retrieved {len(patterns)} active patterns")
    return patterns

  def get_recent_events(self, limit: int = 10) -> List[dict[str, Any]]:
    """Get recent events.

    Args:
      limit: Maximum number of events to return

    Returns:
      List of recent events
    """
    cursor: sqlite3.Cursor = self.conn.cursor()
    cursor.execute(
      """
      SELECT app_name, process_id, started_at, session_id
      FROM events
      ORDER BY started_at DESC
      LIMIT ?
      """,
      (limit,),
    )

    events: List[dict[str, Any]] = []
    for row in cursor.fetchall():
      event: dict[str, Any] = {
        "app_name": row["app_name"],
        "process_id": row["process_id"],
        "started_at": datetime.fromisoformat(row["started_at"]),
        "session_id": row["session_id"],
      }
      events.append(event)

    return events

  def add_feedback(self, pattern_id: int, action: str) -> int:
    """Add user feedback for a pattern.

    Args:
      pattern_id: Pattern ID
      action: User action ('accept', 'reject', 'delete')

    Returns:
      Feedback ID
    """
    cursor: sqlite3.Cursor = self.conn.cursor()
    cursor.execute(
      """
      INSERT INTO feedbacks (pattern_id, action)
      VALUES (?, ?)
      """,
      (pattern_id, action),
    )
    self.conn.commit()
    feedback_id: int = cursor.lastrowid or 0

    # If action is 'delete', deactivate the pattern
    if action == "delete":
      cursor.execute("UPDATE patterns SET is_active = 0 WHERE id = ?", (pattern_id,))
      self.conn.commit()
      logging.info(f"Deactivated pattern {pattern_id} due to user feedback")

    logging.debug(f"Added feedback for pattern {pattern_id}: {action}")
    return feedback_id

  def get_pattern_stats(self) -> dict[str, Any]:
    """Get pattern statistics.

    Returns:
      Dictionary with pattern statistics
    """
    cursor: sqlite3.Cursor = self.conn.cursor()

    # Total patterns
    cursor.execute("SELECT COUNT(*) as count FROM patterns WHERE is_active = 1")
    total_patterns: int = cursor.fetchone()["count"]

    # Total events
    cursor.execute("SELECT COUNT(*) as count FROM events")
    total_events: int = cursor.fetchone()["count"]

    # Average confidence
    cursor.execute("SELECT AVG(confidence) as avg_conf FROM patterns WHERE is_active = 1")
    avg_confidence: Optional[float] = cursor.fetchone()["avg_conf"]

    # Most common apps
    cursor.execute(
      """
      SELECT app_name, COUNT(*) as count
      FROM events
      GROUP BY app_name
      ORDER BY count DESC
      LIMIT 5
      """
    )
    top_apps: List[tuple[str, int]] = [(row["app_name"], row["count"]) for row in cursor.fetchall()]

    return {
      "total_patterns": total_patterns,
      "total_events": total_events,
      "average_confidence": round(avg_confidence or 0.0, 1),
      "top_apps": top_apps,
    }

  def reset_all_data(self) -> None:
    """Clear all learned data from the database using SQL DELETE."""
    cursor: sqlite3.Cursor = self.conn.cursor()
    
    try:
      # Get stats before reset for logging
      stats = self.get_pattern_stats()
      logging.info(f"Resetting database with {stats['total_patterns']} patterns and {stats['total_events']} events")
      
      # Clear all tables
      cursor.execute("DELETE FROM feedbacks")
      cursor.execute("DELETE FROM patterns")
      cursor.execute("DELETE FROM events")
      cursor.execute("DELETE FROM settings")
      
      # Reset autoincrement counters
      cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('feedbacks', 'patterns', 'events')")
      
      self.conn.commit()
      logging.info("All data successfully reset from database")
      
    except Exception as e:
      self.conn.rollback()
      logging.error(f"Error resetting database: {e}")
      raise

  def close(self) -> None:
    """Close database connection."""
    if self.conn:
      self.conn.close()
      logging.info("Database connection closed")
