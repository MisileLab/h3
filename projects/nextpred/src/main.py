"""Main entry point for Next Action Predictor."""

import logging
import signal
import sys
import threading
import time
from typing import Optional

from config import Config
from database import DatabaseManager
from monitor import ProcessMonitor
from pattern_engine import PatternEngine
from notifier import Notifier
from tray_app import TrayApp


def setup_logging(config: Config) -> None:
  """Configure logging system.

  Args:
    config: Configuration object
  """
  log_path = config.get_log_path()
  log_level = config.get_str("Logging", "level", "INFO")

  # Create log directory
  log_path.parent.mkdir(parents=True, exist_ok=True)

  # Configure logging
  logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
  )

  # Reduce noise from some libraries
  logging.getLogger("PIL").setLevel(logging.WARNING)
  logging.getLogger("pystray").setLevel(logging.WARNING)


def signal_handler(signum: int, frame) -> None:
  """Handle shutdown signals.

  Args:
    signum: Signal number
    frame: Current stack frame
  """
  logging.info(f"Received signal {signum}, shutting down...")
  sys.exit(0)


def pattern_analysis_loop(engine: PatternEngine, stop_event: threading.Event) -> None:
  """Background loop for pattern analysis.

  Args:
    engine: Pattern engine instance
    stop_event: Event to signal when to stop
  """
  logging.info("Pattern analysis loop started")

  while not stop_event.is_set():
    try:
      # Analyze patterns every 5 minutes
      engine.analyze_sessions(days=7)

      # Wait for 5 minutes or until stop is requested
      if stop_event.wait(timeout=300):
        break

    except Exception as e:
      logging.error(f"Error in pattern analysis loop: {e}")
      # Wait a shorter time on error
      if stop_event.wait(timeout=60):
        break

  logging.info("Pattern analysis loop stopped")


def suggestion_loop(
  monitor: ProcessMonitor, engine: PatternEngine, notifier: Notifier, stop_event: threading.Event
) -> None:
  """Background loop for checking and showing suggestions.

  Args:
    monitor: Process monitor instance
    engine: Pattern engine instance
    notifier: Notifier instance
    stop_event: Event to signal when to stop
  """
  logging.info("Suggestion loop started")

  while not stop_event.is_set():
    try:
      # Get recent apps
      recent_apps = monitor.get_recent_apps(limit=5)

      if recent_apps:
        # Check for pattern matches
        suggestion = engine.check_pattern_match(recent_apps)

        if suggestion:
          notifier.show_suggestion(
            suggestion["pattern_id"],
            suggestion["next_app"],
            suggestion["confidence"],
            suggestion.get("sequence"),
          )

      # Check every 30 seconds
      if stop_event.wait(timeout=30):
        break

    except Exception as e:
      logging.error(f"Error in suggestion loop: {e}")
      # Wait a shorter time on error
      if stop_event.wait(timeout=10):
        break

  logging.info("Suggestion loop stopped")


def main() -> None:
  """Main entry point for Next Action Predictor."""
  try:
    # Initialize configuration
    config = Config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("Starting Next Action Predictor...")

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize core components
    db = DatabaseManager()

    excluded_processes = config.get_excluded_processes()
    session_timeout = config.get_int("Pattern", "session_timeout_minutes", 5)

    monitor = ProcessMonitor(
      db, excluded_processes=excluded_processes, session_timeout_minutes=session_timeout
    )

    min_occurrences = config.get_int("Pattern", "min_occurrences", 3)
    min_confidence = config.get_float("Pattern", "min_confidence", 60.0)

    engine = PatternEngine(db, min_occurrences=min_occurrences, min_confidence=min_confidence)

    cooldown_hours = config.get_int("Notification", "cooldown_hours", 1)
    notifier = Notifier(db, cooldown_hours=cooldown_hours)

    # Test notification system
    if not notifier.is_notification_available():
      logger.warning("Notification system may not be available on this system")

    # Initialize tray application
    tray_app = TrayApp(monitor, engine, notifier, db)

    # Create stop event for background threads
    stop_event = threading.Event()

    # Start background threads
    monitor_thread = threading.Thread(target=monitor.start, daemon=True)

    analysis_thread = threading.Thread(
      target=pattern_analysis_loop, args=(engine, stop_event), daemon=True
    )

    suggestion_thread = threading.Thread(
      target=suggestion_loop, args=(monitor, engine, notifier, stop_event), daemon=True
    )

    # Start all threads
    monitor_thread.start()
    analysis_thread.start()
    suggestion_thread.start()

    logger.info("All background threads started")

    # Show startup notification
    try:
      stats = db.get_pattern_stats()
      notifier.show_success(
        f"Next Action Predictor started!\n"
        f"Patterns: {stats['total_patterns']}\n"
        f"Events: {stats['total_events']}"
      )
    except Exception as e:
      logger.warning(f"Could not show startup notification: {e}")

    # Start tray application (this blocks)
    try:
      tray_app.start()
    except KeyboardInterrupt:
      logger.info("Received keyboard interrupt")
    except Exception as e:
      logger.error(f"Error in tray application: {e}")
    finally:
      # Clean shutdown
      logger.info("Shutting down...")
      stop_event.set()

      # Stop monitor
      monitor.stop()

      # Wait for threads to finish (with timeout)
      monitor_thread.join(timeout=5)
      analysis_thread.join(timeout=5)
      suggestion_thread.join(timeout=5)

      # Close database
      db.close()

      logger.info("Shutdown complete")

  except KeyboardInterrupt:
    logger.info("Application terminated by user")
    sys.exit(0)
  except Exception as e:
    logger.exception(f"Fatal error: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
