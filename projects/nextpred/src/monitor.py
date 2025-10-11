"""Process monitoring for Next Action Predictor."""

import logging
import platform
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Set, Any

import psutil

from database import DatabaseManager


class ProcessMonitor:
  """Cross-platform process monitor for tracking app execution."""

  def __init__(
    self,
    db_manager: DatabaseManager,
    excluded_processes: Optional[Set[str]] = None,
    session_timeout_minutes: int = 5,
  ) -> None:
    """Initialize process monitor.

    Args:
      db_manager: Database manager instance
      excluded_processes: Set of process names to exclude
      session_timeout_minutes: Minutes of inactivity before starting new session
    """
    self.db: DatabaseManager = db_manager
    self.excluded_processes: Set[str] = excluded_processes or self._get_default_excluded_processes()
    self.session_timeout_minutes: int = session_timeout_minutes

    self.running_processes: Set[int] = set()
    self.current_session_id: Optional[str] = None
    self.last_event_time: Optional[datetime] = None
    self.is_running: bool = False

    logging.info(
      f"Process monitor initialized with {len(self.excluded_processes)} excluded processes"
    )

  def _get_default_excluded_processes(self) -> Set[str]:
    """Get default excluded processes based on platform.

    Returns:
      Set of process names to exclude
    """
    if platform.system() == "Windows":
      return {
        "svchost.exe",
        "System",
        "dwm.exe",
        "explorer.exe",
        "winlogon.exe",
        "csrss.exe",
        "lsass.exe",
        "services.exe",
        "spoolsv.exe",
        "taskhost.exe",
        "audiodg.exe",
      }
    else:
      return {
        "systemd",
        "kthreadd",
        "ksoftirqd",
        "migration",
        "rcu_gp",
        "rcu_par_gp",
        "kworker",
        "init",
        "kswapd",
        "khugepaged",
      }

  def _is_sensitive_process(self, app_name: str) -> bool:
    """Check if process name contains sensitive keywords.

    Args:
      app_name: Process name to check

    Returns:
      True if process appears to be sensitive
    """
    sensitive_keywords: Set[str] = {
      "password",
      "bank",
      "wallet",
      "login",
      "auth",
      "keychain",
      "credential",
      "token",
      "secret",
      "private",
    }

    app_name_lower: str = app_name.lower()
    return any(keyword in app_name_lower for keyword in sensitive_keywords)

  def _should_exclude_process(self, app_name: str) -> bool:
    """Check if process should be excluded from monitoring.

    Args:
      app_name: Process name to check

    Returns:
      True if process should be excluded
    """
    # Check explicit exclusions
    if app_name in self.excluded_processes:
      return True

    # Check sensitive keywords
    if self._is_sensitive_process(app_name):
      logging.debug(f"Excluding sensitive process: {app_name}")
      return True

    # Check if it's a system process (starts with '[' on Linux)
    if app_name.startswith("[") and app_name.endswith("]"):
      return True

    return False

  def _get_session_id(self) -> str:
    """Get current session ID, creating new one if needed.

    Returns:
      Session identifier
    """
    now: datetime = datetime.now()

    # Check if we need a new session
    if (
      self.current_session_id is None
      or self.last_event_time is None
      or now - self.last_event_time > timedelta(minutes=self.session_timeout_minutes)
    ):
      self.current_session_id = str(uuid.uuid4())
      logging.debug(f"Started new session: {self.current_session_id}")

    self.last_event_time = now
    return self.current_session_id

  def _get_process_name(self, proc: psutil.Process) -> Optional[str]:
    """Get process name safely.

    Args:
      proc: Process object

    Returns:
      Process name or None if unable to get it
    """
    try:
      name: str = proc.name()
      # On Windows, include .exe extension for consistency
      if platform.system() == "Windows" and not name.lower().endswith(".exe"):
        return f"{name}.exe"
      return name
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
      return None

  def check_processes(self) -> None:
    """Scan currently running processes and detect new ones."""
    try:
      current_pids: Set[int] = set(psutil.pids())
      new_pids: Set[int] = current_pids - self.running_processes

      for pid in new_pids:
        try:
          proc: psutil.Process = psutil.Process(pid)
          app_name: Optional[str] = self._get_process_name(proc)

          if app_name and not self._should_exclude_process(app_name):
            session_id: str = self._get_session_id()
            self.db.log_event(app_name, pid, session_id)
            logging.info(f"Detected new process: {app_name} (PID: {pid})")

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
          # Process may have ended or we don't have permission
          continue

      # Update running processes set
      self.running_processes = current_pids

      # Clean up dead processes from the set
      dead_pids: Set[int] = self.running_processes - current_pids
      for pid in dead_pids:
        self.running_processes.discard(pid)

    except Exception as e:
      logging.error(f"Error checking processes: {e}")

  def start(self) -> None:
    """Start background process monitoring."""
    self.is_running = True
    logging.info("Starting process monitoring...")

    try:
      while self.is_running:
        self.check_processes()
        time.sleep(1)  # Poll every second
    except KeyboardInterrupt:
      logging.info("Process monitoring stopped by user")
    except Exception as e:
      logging.error(f"Process monitoring error: {e}")
    finally:
      self.is_running = False
      logging.info("Process monitoring stopped")

  def stop(self) -> None:
    """Stop process monitoring."""
    self.is_running = False
    logging.info("Stopping process monitoring...")

  def get_current_session_apps(self) -> list[str]:
    """Get apps in current session.

    Returns:
      List of app names in current session
    """
    if not self.current_session_id:
      return []

    recent_events: list[dict[str, Any]] = self.db.get_recent_events(limit=50)
    session_apps: list[str] = [
      event["app_name"] for event in recent_events if event["session_id"] == self.current_session_id
    ]

    return session_apps

  def get_recent_apps(self, limit: int = 10) -> list[str]:
    """Get recently executed apps.

    Args:
      limit: Maximum number of apps to return

    Returns:
      List of recent app names
    """
    recent_events: list[dict[str, Any]] = self.db.get_recent_events(limit=limit)
    return [event["app_name"] for event in recent_events]

  def is_process_running(self, app_name: str) -> bool:
    """Check if a specific process is currently running.

    Args:
      app_name: Process name to check

    Returns:
      True if process is running
    """
    try:
      for proc in psutil.process_iter(["name"]):
        try:
          proc_name: str = proc.info["name"]
          if proc_name:
            # Normalize comparison
            if platform.system() == "Windows":
              proc_name = proc_name.lower()
              app_name_cmp = app_name.lower()
              if not proc_name.endswith(".exe"):
                proc_name += ".exe"
              if not app_name_cmp.endswith(".exe"):
                app_name_cmp += ".exe"
            else:
              proc_name = proc_name.lower()
              app_name_cmp = app_name.lower()

            if proc_name == app_name_cmp:
              return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
          continue
    except Exception as e:
      logging.error(f"Error checking if process {app_name} is running: {e}")

    return False

  def get_system_info(self) -> dict[str, Any]:
    """Get system information for debugging.

    Returns:
      Dictionary with system information
    """
    return {
      "platform": platform.system(),
      "platform_release": platform.release(),
      "platform_version": platform.version(),
      "architecture": platform.machine(),
      "hostname": platform.node(),
      "processor": platform.processor(),
      "python_version": platform.python_version(),
      "total_processes": len(psutil.pids()),
      "excluded_processes_count": len(self.excluded_processes),
      "current_session_id": self.current_session_id,
      "is_running": self.is_running,
    }
