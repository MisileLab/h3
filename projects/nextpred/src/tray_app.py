"""System tray application for Next Action Predictor."""

import logging
import threading
import time
from typing import Optional, Callable, Any

from PIL import Image, ImageDraw
import pystray

from database import DatabaseManager
from monitor import ProcessMonitor
from pattern_engine import PatternEngine
from notifier import Notifier


class TrayApp:
  """System tray application interface."""

  def __init__(
    self,
    monitor: ProcessMonitor,
    engine: PatternEngine,
    notifier: Notifier,
    db_manager: DatabaseManager,
  ) -> None:
    """Initialize tray application.

    Args:
      monitor: Process monitor instance
      engine: Pattern engine instance
      notifier: Notifier instance
      db_manager: Database manager instance
    """
    self.monitor: ProcessMonitor = monitor
    self.engine: PatternEngine = engine
    self.notifier: Notifier = notifier
    self.db: DatabaseManager = db_manager

    self.icon: Optional[pystray.Icon] = None
    self.is_running: bool = False

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
      pystray.MenuItem("Test Notification", self._test_notification),
      pystray.Menu.SEPARATOR,
      pystray.MenuItem("Analyze Now", self._analyze_patterns),
      pystray.MenuItem("Clear Cooldowns", self._clear_cooldowns),
      pystray.Menu.SEPARATOR,
      pystray.MenuItem("Settings", self._show_settings),
      pystray.MenuItem("Help", self._show_help),
      pystray.Menu.SEPARATOR,
      pystray.MenuItem("Exit", self._exit),
    ]

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
