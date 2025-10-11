"""Cross-platform notification system for Next Action Predictor."""

import logging
import platform
import subprocess
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from database import DatabaseManager


class Notifier:
  """Cross-platform notification manager."""

  def __init__(self, db_manager: DatabaseManager, cooldown_hours: int = 1, tab_callback: Optional[callable] = None) -> None:
      """Initialize notifier.

      Args:
        db_manager: Database manager instance
        cooldown_hours: Cooldown period between notifications for same pattern
        tab_callback: Callback function to call when showing suggestion (for Tab key functionality)
      """
      self.db: DatabaseManager = db_manager
      self.cooldown_hours: int = cooldown_hours
      self.last_notification_time: Dict[int, datetime] = {}
      self.tab_callback: Optional[callable] = tab_callback

      # Try to import plyer for cross-platform notifications
      try:
        from plyer import notification

        self.notification = notification
        self.use_plyer: bool = True
        logging.info("Using plyer for notifications")

        # Force notify-send on Linux per user request
        if platform.system() == "Linux":
          self.use_plyer = False
          logging.info("Forcing notify-send on Linux")
      except ImportError:
        self.notification = None
        self.use_plyer: bool = False
        logging.warning("plyer not available, using fallback notifications")
      logging.info(f"Notifier initialized with {cooldown_hours}h cooldown")

  def _is_recently_notified(self, pattern_id: int) -> bool:
    """Check if notification was recently shown for this pattern.

    Args:
      pattern_id: Pattern ID to check

    Returns:
      True if notification was shown recently
    """
    if pattern_id not in self.last_notification_time:
      return False

    elapsed: timedelta = datetime.now() - self.last_notification_time[pattern_id]
    return elapsed < timedelta(hours=self.cooldown_hours)

  def _send_notification_plyer(self, title: str, message: str, app_name: str) -> bool:
    """Send notification using plyer.

    Args:
      title: Notification title
      message: Notification message
      app_name: Application name for context

    Returns:
      True if notification was sent successfully
    """
    try:
      self.notification.notify(
        title=title,
        message=message,
        app_name=app_name,
        timeout=10,  # 10 seconds
        app_icon=self._get_icon_path(),
      )
      return True
    except Exception as e:
      logging.error(f"Error sending plyer notification: {e}")
      return False

  def _send_notification_fallback(self, title: str, message: str) -> bool:
    """Send notification using platform-specific fallback.

    Args:
      title: Notification title
      message: Notification message

    Returns:
      True if notification was sent successfully
    """
    try:
      if platform.system() == "Windows":
        # Use PowerShell for Windows notifications
        ps_command = f'''
        Add-Type -AssemblyName System.Windows.Forms
        $notification = New-Object System.Windows.Forms.NotifyIcon
        $notification.Icon = [System.Drawing.SystemIcons]::Information
        $notification.BalloonTipTitle = "{title}"
        $notification.BalloonTipText = "{message}"
        $notification.Visible = $true
        $notification.ShowBalloonTip(10000)
        '''
        subprocess.run(["powershell", "-Command", ps_command], capture_output=True, timeout=15)

      elif platform.system() == "Linux":
        # Try notify-send for Linux
        subprocess.run(
          [
            "notify-send",
            "-t",
            "10000",  # 10 seconds
            "-i",
            "dialog-information",
            title,
            message,
          ],
          capture_output=True,
          timeout=15,
        )

      elif platform.system() == "Darwin":  # macOS
        # Use osascript for macOS
        applescript = f'''
        display notification "{message}" with title "{title}"
        '''
        subprocess.run(["osascript", "-e", applescript], capture_output=True, timeout=15)

      return True

    except Exception as e:
      logging.error(f"Error sending fallback notification: {e}")
      return False

  def _get_icon_path(self) -> Optional[str]:
    """Get path to notification icon.

    Returns:
      Path to icon file or None if not found
    """
    import os
    from pathlib import Path

    # Try to find icon in common locations
    icon_paths: list[str] = [
      "resources/icon.png",
      "resources/icon.ico",
      os.path.expanduser("~/.local/share/icons/next-action-predictor.png"),
      "/usr/share/icons/next-action-predictor.png",
    ]

    for path in icon_paths:
      if Path(path).exists():
        return path

    return None

  def show_suggestion(
    self, pattern_id: int, app_name: str, confidence: float, sequence: Optional[list[str]] = None
  ) -> bool:
    """Show next app execution suggestion notification.

    Args:
      pattern_id: Pattern ID
      app_name: Suggested next app name
      confidence: Pattern confidence percentage
      sequence: Full sequence (optional)

    Returns:
      True if notification was shown
    """
    if self._is_recently_notified(pattern_id):
      logging.debug(f"Skipping notification for pattern {pattern_id} - cooldown active")
      return False

    title: str = "🎯 Next Action Predictor"

    # Build message
    message_parts: list[str] = [f"Would you like to run this next?", "", f"📱 {app_name}", ""]

    if sequence and len(sequence) > 1:
      sequence_str: str = " → ".join(sequence[:-1])
      message_parts.append(f"Pattern: {sequence_str} → {app_name}")

    message_parts.append(f"Confidence: {confidence:.0f}%")

    message: str = "\n".join(message_parts)

    # Send notification
    success: bool = False

    if self.use_plyer:
      success = self._send_notification_plyer(title, message, app_name)
    else:
      success = self._send_notification_fallback(title, message)

    if success:
      self.last_notification_time[pattern_id] = datetime.now()
      logging.info(
        f"Shown suggestion for {app_name} (pattern {pattern_id}, confidence: {confidence:.0f}%)"
      )

      # Call Tab callback if available
      if self.tab_callback:
        try:
          self.tab_callback(app_name)
        except Exception as e:
          logging.error(f"Error calling Tab callback: {e}")

      # Log the notification as a feedback event
      self.db.add_feedback(pattern_id, "shown")
    else:
      logging.error(f"Failed to show suggestion for {app_name}")

    return success

  def show_learning_status(self, patterns_count: int, events_count: int) -> bool:
    """Show learning status notification.

    Args:
      patterns_count: Number of learned patterns
      events_count: Number of recorded events

    Returns:
      True if notification was shown successfully
    """
    title: str = "📊 Next Action Predictor Status"
    message: str = f"Learning Progress:\n\n🧠 {patterns_count} patterns learned\n📈 {events_count} events recorded"

    if self.use_plyer:
      return self._send_notification_plyer(title, message, "Next Action Predictor")
    else:
      return self._send_notification_fallback(title, message)

  def show_error(self, error_message: str) -> bool:
    """Show error notification.

    Args:
      error_message: Error message to display

    Returns:
      True if notification was shown successfully
    """
    title: str = "❌ Next Action Predictor Error"
    message: str = f"An error occurred:\n\n{error_message}"

    if self.use_plyer:
      return self._send_notification_plyer(title, message, "Next Action Predictor")
    else:
      return self._send_notification_fallback(title, message)

  def show_success(self, success_message: str) -> bool:
    """Show success notification.

    Args:
      success_message: Success message to display

    Returns:
      True if notification was shown successfully
    """
    title: str = "✅ Next Action Predictor"
    message: str = success_message

    if self.use_plyer:
      return self._send_notification_plyer(title, message, "Next Action Predictor")
    else:
      return self._send_notification_fallback(title, message)

  def test_notification(self) -> bool:
    """Send a test notification.

    Returns:
      True if test notification was successful
    """
    title: str = "🧪 Test Notification"
    message: str = "Next Action Predictor is working correctly!"

    if self.use_plyer:
      success = self._send_notification_plyer(title, message, "Next Action Predictor")
    else:
      success = self._send_notification_fallback(title, message)

    if success:
      logging.info("Test notification sent successfully")
    else:
      logging.error("Test notification failed")

    return success

  def clear_cooldown(self, pattern_id: Optional[int] = None) -> None:
    """Clear notification cooldown.

    Args:
      pattern_id: Specific pattern ID to clear, or None to clear all
    """
    if pattern_id:
      if pattern_id in self.last_notification_time:
        del self.last_notification_time[pattern_id]
        logging.debug(f"Cleared cooldown for pattern {pattern_id}")
    else:
      self.last_notification_time.clear()
      logging.debug("Cleared all notification cooldowns")

  def get_notification_stats(self) -> Dict[str, Any]:
    """Get notification statistics.

    Returns:
      Dictionary with notification statistics
    """
    total_notifications: int = len(self.last_notification_time)
    recent_notifications: int = sum(
      1
      for timestamp in self.last_notification_time.values()
      if datetime.now() - timestamp < timedelta(hours=24)
    )

    return {
      "total_patterns_notified": total_notifications,
      "recent_notifications_24h": recent_notifications,
      "cooldown_hours": self.cooldown_hours,
      "using_plyer": self.use_plyer,
    }

  def is_notification_available(self) -> bool:
    """Check if notifications are available on this system.

    Returns:
      True if notifications should work
    """
    if self.use_plyer:
      return True

    # Check fallback availability
    if platform.system() == "Linux":
      try:
        subprocess.run(["which", "notify-send"], capture_output=True, check=True)
        return True
      except subprocess.CalledProcessError:
        return False

    elif platform.system() == "Darwin":
      try:
        subprocess.run(["which", "osascript"], capture_output=True, check=True)
        return True
      except subprocess.CalledProcessError:
        return False

    elif platform.system() == "Windows":
      # PowerShell should always be available on Windows
      return True

    return False
