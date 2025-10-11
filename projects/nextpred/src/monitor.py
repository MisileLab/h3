"""Process monitoring for Next Action Predictor."""

import logging
import platform
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Set, Any, Tuple

import psutil

from database import DatabaseManager

# Platform-specific imports for window tracking
WIN32_AVAILABLE = False
MACOS_AVAILABLE = False
LINUX_AVAILABLE = False

if platform.system() == "Windows":
    try:
        import win32gui
        import win32process
        import win32api
        import win32con
        WIN32_AVAILABLE = True
    except ImportError:
        WIN32_AVAILABLE = False
        win32gui = None
        win32process = None
        win32api = None
        win32con = None
elif platform.system() == "Darwin":  # macOS
    try:
        # type: ignore
        from AppKit import NSWorkspace
        from Quartz import CGEventCreate, CGEventGetLocation, kCGEventSourceStateHIDSystemState
        MACOS_AVAILABLE = True
    except ImportError:
        MACOS_AVAILABLE = False
        NSWorkspace = None
else:  # Linux
    try:
        # type: ignore
        import ewmh
        # type: ignore
        import Xlib.display
        from Xlib import X
        LINUX_AVAILABLE = True
    except ImportError:
        LINUX_AVAILABLE = False
        ewmh = None
        Xlib = None


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
    
    # Window tracking
    self.last_focused_window: Optional[str] = None
    self.last_window_check: Optional[datetime] = None

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

    # Additional system processes to exclude
    system_processes = {
      'System Idle Process.exe',
      'System.exe',
      'svchost.exe',
      'dwm.exe',
      'explorer.exe',
      'winlogon.exe',
      'csrss.exe',
      'lsass.exe',
      'services.exe',
      'spoolsv.exe',
      'taskhost.exe',
      'audiodg.exe',
      'sihost.exe',
      'fontdrvhost.exe',
      'Registry.exe',
      'RuntimeBroker.exe',
      'ApplicationFrameHost.exe',
      'ShellHost.exe',
      'SearchHost.exe',
      'StartMenuExperienceHost.exe',
      'TextInputHost.exe',
      'Widgets.exe',
      'SystemSettings.exe',
      'SecurityHealthSystray.exe',
      'backgroundTaskHost.exe',
      'PhoneExperienceHost.exe',
      'CrossDeviceService.exe',
      'vmms.exe',
      'WUDFHost.exe',
      'MemCompression.exe',
      'smartscreen.exe',
      'unsecapp.exe',
      'dllhost.exe',
      'conhost.exe',
      'ctfmon.exe',
      'rundll32.exe',
    }
    
    if app_name in system_processes:
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

  def _get_focused_window_app(self) -> Optional[str]:
    """Get the application name of the currently focused window.

    Returns:
      Application name or None if unable to get it
    """
    try:
      if platform.system() == "Windows" and WIN32_AVAILABLE:
        return self._get_focused_window_windows()
      elif platform.system() == "Darwin" and MACOS_AVAILABLE:
        return self._get_focused_window_macos()
      elif platform.system() == "Linux" and LINUX_AVAILABLE:
        return self._get_focused_window_linux()
      else:
        # Fallback to process-based detection
        return self._get_focused_window_fallback()
    except Exception as e:
      logging.debug(f"Error getting focused window: {e}")
      return None

  def _get_focused_window_windows(self) -> Optional[str]:
    """Get focused window on Windows."""
    try:
      if not win32gui or not win32process:
        return None
        
      hwnd = win32gui.GetForegroundWindow()
      _, pid = win32process.GetWindowThreadProcessId(hwnd)
      
      proc = psutil.Process(pid)
      app_name = self._get_process_name(proc)
      
      if app_name and not self._should_exclude_process(app_name):
        return app_name
      return None
    except Exception as e:
      logging.debug(f"Error getting Windows focused window: {e}")
      return None

  def _get_focused_window_macos(self) -> Optional[str]:
    """Get focused window on macOS."""
    try:
      if not NSWorkspace:
        return None
        
      workspace = NSWorkspace.sharedWorkspace()
      active_app = workspace.activeApplication()
      
      if active_app:
        app_name = active_app['NSApplicationName']
        # Add .exe for consistency if needed
        if platform.system() == "Windows" and not app_name.lower().endswith(".exe"):
          app_name = f"{app_name}.exe"
        
        if not self._should_exclude_process(app_name):
          return app_name
      return None
    except Exception as e:
      logging.debug(f"Error getting macOS focused window: {e}")
      return None

  def _get_focused_window_linux(self) -> Optional[str]:
    """Get focused window on Linux."""
    try:
      if not ewmh or not Xlib:
        return None
        
      display = Xlib.display.Display()
      root = display.screen().root
      ewmh_obj = ewmh.EWMH(root, display)
      
      window = ewmh_obj.getActiveWindow()
      if window:
        pid = window.get_wm_pid()
        if pid:
          proc = psutil.Process(pid)
          app_name = self._get_process_name(proc)
          
          if app_name and not self._should_exclude_process(app_name):
            return app_name
      return None
    except Exception as e:
      logging.debug(f"Error getting Linux focused window: {e}")
      return None

  def _get_focused_window_fallback(self) -> Optional[str]:
    """Fallback method using most recently active process."""
    try:
      # Get all processes and sort by CPU time (approximation of activity)
      processes = []
      for proc in psutil.process_iter(['name', 'cpu_times']):
        try:
          cpu_times = proc.info['cpu_times']
          if cpu_times and cpu_times.user > 0:
            processes.append((proc, cpu_times.user))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
          continue
      
      if processes:
        # Sort by CPU time and get the most active
        processes.sort(key=lambda x: x[1], reverse=True)
        most_active_proc = processes[0][0]
        app_name = self._get_process_name(most_active_proc)
        
        if app_name and not self._should_exclude_process(app_name):
          return app_name
      return None
    except Exception as e:
      logging.debug(f"Error in fallback focused window detection: {e}")
      return None

  def check_processes(self) -> None:
    """Check currently visible applications and detect changes."""
    try:
      now = datetime.now()
      
      # Get all visible applications
      visible_apps = self._get_visible_applications()
      
      if visible_apps:
        # Get the most recently active app (first in list)
        current_app = visible_apps[0]
        
        # Check if this is a different app than last time
        if current_app != self.last_focused_window:
          session_id: str = self._get_session_id()
          
          # Try to get PID for the current app
          pid = self._get_app_pid(current_app)
          
          self.db.log_event(current_app, pid, session_id)
          logging.info(f"Active app changed to: {current_app} (PID: {pid})")
          
          self.last_focused_window = current_app
          self.last_window_check = now
      
      # Update running processes set (for compatibility)
      current_pids: Set[int] = set(psutil.pids())
      self.running_processes = current_pids

    except Exception as e:
      logging.error(f"Error checking processes: {e}")

  def _get_visible_applications(self) -> list[str]:
    """Get list of currently visible applications.
    
    Returns:
      List of application names that are visible to the user
    """
    try:
      if platform.system() == "Windows" and WIN32_AVAILABLE:
        return self._get_visible_applications_windows()
      elif platform.system() == "Darwin" and MACOS_AVAILABLE:
        return self._get_visible_applications_macos()
      elif platform.system() == "Linux" and LINUX_AVAILABLE:
        return self._get_visible_applications_linux()
      else:
        # Fallback to process-based detection
        return self._get_visible_applications_fallback()
    except Exception as e:
      logging.debug(f"Error getting visible applications: {e}")
      return []

  def _get_visible_applications_windows(self) -> list[str]:
    """Get visible applications on Windows using EnumWindows."""
    try:
      if not win32gui or not win32process:
        return []
      
      visible_apps = []
      
      def enum_windows_callback(hwnd, apps):
        try:
          # Check if window is visible and has a title
          if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if window_title and len(window_title.strip()) > 0:
              # Get window style to filter out system windows
              try:
                style = win32gui.GetWindowLong(hwnd, 0)  # GWL_STYLE
                ex_style = win32gui.GetWindowLong(hwnd, -20)  # GWL_EXSTYLE
                
                # Skip certain window types
                if (ex_style & 0x00000080):  # WS_EX_TOOLWINDOW
                  return True
              except:
                # If we can't get window style, continue anyway
                pass
              
              # Get process ID
              _, pid = win32process.GetWindowThreadProcessId(hwnd)
              
              # Get process name
              try:
                if pid > 0:  # Skip PID 0 (System Idle Process)
                  proc = psutil.Process(pid)
                  app_name = self._get_process_name(proc)
                  
                  # Additional validation for app name
                  if (app_name and 
                      len(app_name.strip()) > 1 and  # Skip single character names
                      not app_name.startswith('.') and  # Skip names starting with dot
                      (not app_name.endswith('.exe') or len(app_name) > 5) and  # Allow proper .exe files
                      not self._should_exclude_process(app_name) and 
                      app_name not in apps):
                    apps.append(app_name)
              except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        except Exception:
          pass
        return True
      
      # Enumerate all windows
      win32gui.EnumWindows(enum_windows_callback, visible_apps)
      
      return visible_apps
    except Exception as e:
      logging.debug(f"Error getting Windows visible applications: {e}")
      return []

  def _get_visible_applications_macos(self) -> list[str]:
    """Get visible applications on macOS."""
    try:
      if not NSWorkspace:
        return []
        
      workspace = NSWorkspace.sharedWorkspace()
      running_apps = workspace.runningApplications()
      
      visible_apps = []
      for app in running_apps:
        try:
          # Check if app is visible and has a UI
          if app.activationPolicy() == 0:  # NSApplicationActivationPolicyRegular
            app_name = app.localizedName()
            if app_name and not self._should_exclude_process(app_name):
              visible_apps.append(app_name)
        except Exception:
          continue
      
      return visible_apps
    except Exception as e:
      logging.debug(f"Error getting macOS visible applications: {e}")
      return []

  def _get_visible_applications_linux(self) -> list[str]:
    """Get visible applications on Linux."""
    try:
      if not ewmh or not Xlib:
        return []
        
      display = Xlib.display.Display()
      root = display.screen().root
      ewmh_obj = ewmh.EWMH(root, display)
      
      visible_apps = []
      client_list = ewmh_obj.getClientList()
      
      for window in client_list:
        try:
          # Check if window is mapped (visible)
          if window.get_wm_state() and '_NET_WM_STATE_HIDDEN' not in window.get_wm_state():
            pid = window.get_wm_pid()
            if pid:
              proc = psutil.Process(pid)
              app_name = self._get_process_name(proc)
              
              if app_name and not self._should_exclude_process(app_name) and app_name not in visible_apps:
                visible_apps.append(app_name)
        except Exception:
          continue
      
      return visible_apps
    except Exception as e:
      logging.debug(f"Error getting Linux visible applications: {e}")
      return []

  def _get_visible_applications_fallback(self) -> list[str]:
    """Fallback method using process information."""
    try:
      visible_apps = []
      
      # Get all processes and filter by those with windows
      for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
        try:
          # Filter for processes that are likely to have UI
          cpu_percent = proc.info.get('cpu_percent', 0)
          memory_percent = proc.info.get('memory_percent', 0)
          
          # Skip system processes (low CPU/memory usage)
          if cpu_percent > 0.1 or memory_percent > 0.1:
            app_name = self._get_process_name(proc)
            if app_name and not self._should_exclude_process(app_name) and app_name not in visible_apps:
              visible_apps.append(app_name)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
          continue
      
      return visible_apps
    except Exception as e:
      logging.debug(f"Error in fallback visible applications detection: {e}")
      return []

  def _get_app_pid(self, app_name: str) -> int:
    """Get PID for a given app name.
    
    Args:
      app_name: Application name to find PID for
      
    Returns:
      Process ID or 0 if not found
    """
    try:
      for proc in psutil.process_iter(['pid', 'name']):
        try:
          proc_name = proc.info['name']
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
              return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
          continue
    except Exception as e:
      logging.debug(f"Error getting PID for {app_name}: {e}")
    
    return 0

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
