"""System tray application for Next Action Predictor."""

import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional

import pystray
from PIL import Image, ImageDraw

from config import Config
from database import DatabaseManager
from monitor import ProcessMonitor
from notifier import Notifier
from pattern_engine import PatternEngine
from windows_api import focus_process_by_name


class TrayApp:
  """System tray application interface."""

  def __init__(
    self,
    config: Config,
    monitor: ProcessMonitor,
    engine: PatternEngine,
    notifier: Notifier,
    db_manager: DatabaseManager,
  ) -> None:
    """Initialize tray application.

    Args:
      config: Configuration object
      monitor: Process monitor instance
      engine: Pattern engine instance
      notifier: Notifier instance
      db_manager: Database manager instance
    """
    self.config = config
    self.monitor = monitor
    self.engine = engine
    self.notifier = notifier
    self.db = db_manager

    self.icon = None
    self.is_running = False
    
    # Tab key functionality
    self.last_suggested_process = None
    self.tab_listener_thread = None
    self.tab_listener_running = False

    logging.info("Tray application initialized")

  def _create_icon_image(self, color: str = "#4CAF50", size: int = 64) -> Image.Image:
    """Create a simple icon image.

    Args:
      color: Icon color
      size: Icon size in pixels

    Returns:
      PIL Image object
    """
    # Create a simple circular icon with "N" letter
    image: Image.Image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)

    # Draw circle
    margin: int = size // 8
    draw.ellipse([margin, margin, size - margin, size - margin], fill=color)

    # Draw "N" letter
    font_size: int = size // 2
    text_margin: int = size // 6

    # Simple "N" using lines
    n_size: int = size - 2 * text_margin
    line_width: int = max(2, size // 16)

    # Left vertical line
    draw.line(
      [text_margin, text_margin, text_margin, size - text_margin], fill="white", width=line_width
    )

    # Right vertical line
    draw.line(
      [size - text_margin, text_margin, size - text_margin, size - text_margin],
      fill="white",
      width=line_width,
    )

    # Diagonal line
    draw.line(
      [text_margin, size - text_margin, size - text_margin, text_margin],
      fill="white",
      width=line_width,
    )

    return image

  def _get_menu_items(self) -> list[pystray.MenuItem]:
    """Get menu items for the tray icon.

    Returns:
      List of menu items
    """
    return [
      pystray.MenuItem("Learning Status", self._show_status),
      pystray.Menu.SEPARATOR,
      pystray.MenuItem("View Statistics", self._show_stats),
      pystray.MenuItem("Recent Patterns", self._show_patterns),
      pystray.MenuItem("View Logs", self._view_logs),
      pystray.MenuItem("Test Notification", self._test_notification),
      pystray.Menu.SEPARATOR,
      pystray.MenuItem("Analyze Now", self._analyze_patterns),
      pystray.MenuItem("Clear Cooldowns", self._clear_cooldowns),
      pystray.Menu.SEPARATOR,
      pystray.MenuItem("Reset All Data", self._reset_data),
      pystray.Menu.SEPARATOR,
      pystray.MenuItem("Settings", self._show_settings),
      pystray.MenuItem("Help", self._show_help),
      pystray.Menu.SEPARATOR,
      pystray.MenuItem("Exit", self._exit),
    ]

  def _view_logs(self, icon: Optional[Any], item: Optional[Any]) -> None:
    """Open the log file with the default application.

    Args:
      icon: Tray icon (unused)
      item: Menu item (unused)
    """
    log_path = self.config.get_log_path()
    logging.info(f"Opening log file: {log_path}")

    try:
      if not log_path.exists():
        self.notifier.show_error(f"Log file not found at {log_path}")
        logging.warning(f"Log file not found: {log_path}")
        return

      if sys.platform == "win32":
        os.startfile(log_path)
      elif sys.platform == "darwin":
        subprocess.call(["open", log_path])
      else:
        subprocess.call(["xdg-open", log_path])

    except Exception as e:
      logging.error(f"Error opening log file: {e}")
      self.notifier.show_error(f"Failed to open log file: {e}")

  def _show_status(self, icon: Optional[Any], item: Optional[Any]) -> None:
    """Show learning status notification.

    Args:
      icon: Tray icon (unused)
      item: Menu item (unused)
    """
    try:
      stats: dict[str, any] = self.db.get_pattern_stats()
      self.notifier.show_learning_status(stats["total_patterns"], stats["total_events"])
    except Exception as e:
      logging.error(f"Error showing status: {e}")
      self.notifier.show_error(f"Failed to show status: {e}")

  def _show_stats(self, icon: Optional[Any], item: Optional[Any]) -> None:
    """Show detailed statistics.

    Args:
      icon: Tray icon (unused)
      item: Menu item (unused)
    """
    try:
      pattern_stats: dict[str, any] = self.engine.get_pattern_statistics()
      db_stats: dict[str, any] = self.db.get_pattern_stats()
      notification_stats: dict[str, any] = self.notifier.get_notification_stats()

      # Create a detailed statistics message
      message_parts: list[str] = [
        "📊 Next Action Predictor Statistics",
        "",
        "🧠 Patterns:",
        f"  Total: {db_stats['total_patterns']}",
        f"  Average Confidence: {db_stats['average_confidence']}%",
        "",
        "📈 Events:",
        f"  Total Recorded: {db_stats['total_events']}",
        "",
        "🔔 Notifications:",
        f"  Patterns Notified: {notification_stats['total_patterns_notified']}",
        f"  Recent (24h): {notification_stats['recent_notifications_24h']}",
        "",
        "🏆 Top Apps:",
      ]

      for app, count in db_stats["top_apps"][:5]:
        message_parts.append(f"  {app}: {count}")

      message: str = "\n".join(message_parts)

      # Show as notification (truncated if needed)
      if len(message) > 500:
        message = message[:497] + "..."

      self.notifier.show_success("Statistics - Check logs for full details")
      logging.info(f"Statistics:\n{message}")

    except Exception as e:
      logging.error(f"Error showing statistics: {e}")
      self.notifier.show_error(f"Failed to show statistics: {e}")

  def _show_patterns(self, icon: Optional[Any], item: Optional[Any]) -> None:
    """Show recent patterns.

    Args:
      icon: Tray icon (unused)
      item: Menu item (unused)
    """
    try:
      patterns: list[dict[str, any]] = self.engine.get_pattern_suggestions(limit=10)

      if not patterns:
        self.notifier.show_success("No patterns learned yet")
        return

      message_parts: list[str] = ["🧠 Recent Patterns:", ""]

      for i, pattern in enumerate(patterns[:5], 1):
        sequence_str: str = " → ".join(pattern["sequence"])
        message_parts.append(f"{i}. {sequence_str}")
        message_parts.append(f"   Confidence: {pattern['confidence']:.0f}%")
        message_parts.append("")

      message: str = "\n".join(message_parts)
      logging.info(f"Recent patterns:\n{message}")

      self.notifier.show_success(f"Found {len(patterns)} patterns - Check logs")

    except Exception as e:
      logging.error(f"Error showing patterns: {e}")
      self.notifier.show_error(f"Failed to show patterns: {e}")

  def _test_notification(self, icon: Optional[Any], item: Optional[Any]) -> None:
    """Test notification system.

    Args:
      icon: Tray icon (unused)
      item: Menu item (unused)
    """
    try:
      success: bool = self.notifier.test_notification()
      if success:
        self.notifier.show_success("Test notification sent successfully!")
      else:
        self.notifier.show_error("Test notification failed")
    except Exception as e:
      logging.error(f"Error testing notification: {e}")
      self.notifier.show_error(f"Test failed: {e}")

  def _analyze_patterns(self, icon: Optional[Any], item: Optional[Any]) -> None:
    """Manually trigger pattern analysis.

    Args:
      icon: Tray icon (unused)
      item: Menu item (unused)
    """
    try:
      self.notifier.show_success("Starting pattern analysis...")

      # Run analysis in background thread to avoid blocking
      def analyze():
        try:
          self.engine.analyze_sessions(days=7)
          self.notifier.show_success("Pattern analysis completed!")
        except Exception as e:
          logging.error(f"Error in pattern analysis: {e}")
          self.notifier.show_error(f"Analysis failed: {e}")

      threading.Thread(target=analyze, daemon=True).start()

    except Exception as e:
      logging.error(f"Error starting analysis: {e}")
      self.notifier.show_error(f"Failed to start analysis: {e}")

  def _clear_cooldowns(self, icon: Optional[Any], item: Optional[Any]) -> None:
    """Clear all notification cooldowns.

    Args:
      icon: Tray icon (unused)
      item: Menu item (unused)
    """
    try:
      self.notifier.clear_cooldown()
      self.notifier.show_success("All notification cooldowns cleared")
      logging.info("User cleared all notification cooldowns")
    except Exception as e:
      logging.error(f"Error clearing cooldowns: {e}")
      self.notifier.show_error(f"Failed to clear cooldowns: {e}")

  def _show_settings(self, icon: Optional[Any], item: Optional[Any]) -> None:
    """Show settings information.

    Args:
      icon: Tray icon (unused)
      item: Menu item (unused)
    """
    try:
      message_parts: list[str] = [
        "⚙️ Current Settings:",
        "",
        f"Pattern Min Occurrences: {self.engine.min_occurrences}",
        f"Pattern Min Confidence: {self.engine.min_confidence}%",
        f"Session Timeout: {self.monitor.session_timeout_minutes} minutes",
        f"Notification Cooldown: {self.notifier.cooldown_hours} hours",
        "",
        "To change settings, edit the configuration file",
      ]

      message: str = "\n".join(message_parts)
      logging.info(f"Settings:\n{message}")

      self.notifier.show_success("Settings - Check logs for details")

    except Exception as e:
      logging.error(f"Error showing settings: {e}")
      self.notifier.show_error(f"Failed to show settings: {e}")

  def _reset_data(self, icon: Optional[Any], item: Optional[Any]) -> None:
    """Reset all learned data.

    Args:
      icon: Tray icon (unused)
      item: Menu item (unused)
    """
    try:
      # Show confirmation dialog first
      import tkinter as tk
      from tkinter import messagebox, ttk
      
      root = tk.Tk()
      root.withdraw()  # Hide the main window
      root.attributes('-topmost', True)
      
      result = messagebox.askyesno(
        "Confirm Reset",
        "Are you sure you want to reset all learned data?\n\n"
        "This will delete:\n"
        "• All learned patterns\n"
        "• All usage events\n"
        "• All feedback data\n\n"
        "This action cannot be undone!",
        icon='warning'
      )
      
      root.destroy()
      
      if result:
        # Show progress dialog
        progress_root = tk.Tk()
        progress_root.title("Resetting Data")
        progress_root.geometry("300x100")
        progress_root.resizable(False, False)
        progress_root.attributes('-topmost', True)
        
        # Center the window
        progress_root.update_idletasks()
        width = progress_root.winfo_width()
        height = progress_root.winfo_height()
        x = (progress_root.winfo_screenwidth() // 2) - (width // 2)
        y = (progress_root.winfo_screenheight() // 2) - (height // 2)
        progress_root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Create progress bar
        label = tk.Label(progress_root, text="Resetting all learned data...")
        label.pack(pady=10)
        
        # Use simple progress bar instead of ttk
        progress = tk.Canvas(progress_root, height=20, bg='white')
        progress.pack(pady=10, padx=20, fill='x')
        
        # Create animated progress bar
        progress_width = 260
        progress.create_rectangle(0, 0, progress_width, 20, outline='gray', fill='lightgray')
        
        # Animation variables
        progress_pos = 0
        progress_rect = None
        animation_running = True
        
        def animate_progress():
          nonlocal progress_pos, progress_rect
          if not animation_running:
            return
            
          if progress_rect:
            progress.delete(progress_rect)
          
          # Create moving rectangle
          progress_rect = progress.create_rectangle(
            progress_pos, 2, progress_pos + 30, 18, 
            fill='#4CAF50', outline=''
          )
          
          progress_pos += 5
          if progress_pos > progress_width:
            progress_pos = -30
          
          if animation_running:
            progress_root.after(50, animate_progress)
        
        # Start animation
        animate_progress()
        
        # Update UI
        progress_root.update()
        
        def reset_in_background():
          try:
            logging.info("Starting database reset...")
            self.db.reset_all_data()
            logging.info("Database reset completed successfully")
            # Stop animation and close window
            progress_root.after(0, stop_progress_and_close)
            self.notifier.show_success("All data has been reset successfully!")
            logging.info("User reset all learned data")
          except Exception as e:
            logging.error(f"Error resetting data: {e}")
            # Stop animation and close window
            progress_root.after(0, stop_progress_and_close)
            self.notifier.show_error(f"Failed to reset data: {e}")
        
        def stop_progress_and_close():
          nonlocal animation_running
          animation_running = False
          progress_root.destroy()
        
        # Run reset in background thread
        import threading
        threading.Thread(target=reset_in_background, daemon=True).start()
        
        # Show progress dialog
        try:
          progress_root.mainloop()
        except KeyboardInterrupt:
          progress_root.destroy()
      else:
        logging.info("User cancelled data reset")
        
    except Exception as e:
      logging.error(f"Error resetting data: {e}")
      self.notifier.show_error(f"Failed to reset data: {e}")

  def _show_help(self, icon: Optional[Any], item: Optional[Any]) -> None:
    """Show help information.

    Args:
      icon: Tray icon (unused)
      item: Menu item (unused)
    """
    try:
      help_text: str = (
        "🤖 Next Action Predictor Help\n\n"
        "This app learns your app usage patterns and suggests next actions.\n\n"
        "• Patterns are learned automatically\n"
        "• Suggestions appear as notifications\n"
        "• Right-click tray icon for options\n\n"
        "Check the logs for detailed information."
      )

      self.notifier.show_success("Help - Check logs for details")
      logging.info(f"Help:\n{help_text}")

    except Exception as e:
      logging.error(f"Error showing help: {e}")
      self.notifier.show_error(f"Failed to show help: {e}")

  def _exit(self, icon: Optional[Any], item: Optional[Any]) -> None:
    """Exit the application.

    Args:
      icon: Tray icon
      item: Menu item (unused)
    """
    logging.info("User requested exit")
    self.stop()

  def _update_tooltip(self) -> None:
    """Update the tray icon tooltip."""
    if not self.icon:
      return

    try:
      stats: dict[str, any] = self.db.get_pattern_stats()
      tooltip: str = (
        f"Next Action Predictor\n"
        f"Patterns: {stats['total_patterns']}\n"
        f"Events: {stats['total_events']}\n"
        f"Status: {'Running' if self.is_running else 'Stopped'}"
      )
      self.icon.tooltip = tooltip
    except Exception as e:
      logging.error(f"Error updating tooltip: {e}")

  def start(self) -> None:
    """Start the tray application."""
    try:
      self.is_running = True

      # Create icon
      icon_image: Image.Image = self._create_icon_image()
      menu: pystray.Menu = pystray.Menu(*self._get_menu_items())

      self.icon = pystray.Icon("next_action_predictor", icon_image, "Next Action Predictor", menu)

      # Start tooltip update thread
      def update_tooltip_loop():
        while self.is_running:
          try:
            self._update_tooltip()
            time.sleep(30)  # Update every 30 seconds
          except Exception as e:
            logging.error(f"Error in tooltip update loop: {e}")
            time.sleep(5)

      tooltip_thread: threading.Thread = threading.Thread(target=update_tooltip_loop, daemon=True)
      tooltip_thread.start()

      logging.info("Starting tray application...")
      if self.icon:
        self.icon.run()

    except Exception as e:
      logging.error(f"Error starting tray application: {e}")
      self.notifier.show_error(f"Failed to start tray app: {e}")

  def stop(self) -> None:
    """Stop the tray application."""
    try:
      self.is_running = False

      if self.icon:
        self.icon.stop()
        logging.info("Tray application stopped")

      # Stop other components
      if self.monitor:
        self.monitor.stop()

      if self.db:
        self.db.close()

    except Exception as e:
      logging.error(f"Error stopping tray application: {e}")

  def update_status(self, message: str) -> None:
    """Update status message.

    Args:
      message: Status message
    """
    if self.icon:
      self.icon.tooltip = f"Next Action Predictor\n{message}"

    logging.info(f"Status: {message}")

  def set_last_suggested_process(self, process_name: str) -> None:
    """Store the last suggested process for Tab key switching.
    
    Args:
      process_name: Name of the suggested process
    """
    self.last_suggested_process = process_name
    logging.info(f"Last suggested process set to: {process_name}")
    
    # Start Tab listener if not already running
    if not self.tab_listener_running:
      self._start_tab_listener()

  def _start_tab_listener(self) -> None:
    """Start the Tab key listener thread."""
    if self.tab_listener_running:
      return
      
    self.tab_listener_running = True
    self.tab_listener_thread = threading.Thread(target=self._tab_listener_loop, daemon=True)
    self.tab_listener_thread.start()
    logging.info("Tab key listener started")

  def _stop_tab_listener(self) -> None:
    """Stop the Tab key listener thread."""
    self.tab_listener_running = False
    if self.tab_listener_thread:
      self.tab_listener_thread.join(timeout=1.0)
    logging.info("Tab key listener stopped")

  def _tab_listener_loop(self) -> None:
    """Main loop for Tab key listener."""
    try:
      import keyboard
      
      while self.tab_listener_running:
        # Check for Tab key press
        if keyboard.is_pressed('tab'):
          if self.last_suggested_process:
            logging.info(f"Tab key pressed, switching to {self.last_suggested_process}")
            success = focus_process_by_name(self.last_suggested_process)
            if success:
              logging.info(f"Successfully switched to {self.last_suggested_process}")
              # Clear the suggestion after successful switch
              self.last_suggested_process = None
            else:
              logging.warning(f"Failed to switch to {self.last_suggested_process}")
          else:
            logging.debug("Tab key pressed but no suggested process available")
          
          # Wait a bit to prevent multiple detections
          time.sleep(0.5)
        
        time.sleep(0.1)  # Check every 100ms
        
    except ImportError:
      logging.warning("keyboard module not available, Tab key functionality disabled")
    except Exception as e:
      logging.error(f"Error in Tab listener loop: {e}")

  def cleanup(self) -> None:
    """Clean up resources."""
    self._stop_tab_listener()
    logging.info("Tray app cleanup completed")
