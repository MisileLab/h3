"""Configuration management for Next Action Predictor."""

import configparser
import logging
import platform
from pathlib import Path
from typing import Optional, Set, TypedDict, Any


# Define ConfigDict locally to avoid import issues
class ConfigDict(TypedDict):
  """Type for configuration dictionary."""

  min_occurrences: int
  min_confidence: float
  session_timeout_minutes: int
  notification_cooldown_hours: int
  excluded_processes: set[str]


class Config:
  """Configuration manager for Next Action Predictor."""

  def __init__(self, config_path: Optional[str] = None) -> None:
    """Initialize configuration manager.

    Args:
      config_path: Path to config file. If None, uses default location.
    """
    self.config: configparser.ConfigParser = configparser.ConfigParser()

    if config_path:
      self.config_path: Path = Path(config_path)
    else:
      # Use platform-specific config directory
      if platform.system() == "Windows":
        config_dir = Path.home() / "AppData" / "Local" / "NextActionPredictor"
      else:
        config_dir = Path.home() / ".config" / "next-action-predictor"

      config_dir.mkdir(parents=True, exist_ok=True)
      self.config_path = config_dir / "config.ini"

    if self.config_path.exists():
      self.config.read(self.config_path)
      logging.info(f"Loaded configuration from {self.config_path}")
    else:
      self._create_default_config()
      logging.info(f"Created default configuration at {self.config_path}")

  def _create_default_config(self) -> None:
    """Create default configuration file."""
    # Default excluded processes for different platforms
    if platform.system() == "Windows":
      default_excluded = "svchost.exe,System,dwm.exe,explorer.exe,winlogon.exe,csrss.exe,lsass.exe"
    else:
      default_excluded = "systemd,kernelinit,kthreadd,ksoftirqd,migration,rcu_gp,rcu_par_gp"

    self.config["Pattern"] = {
      "min_occurrences": "3",
      "min_confidence": "60.0",
      "session_timeout_minutes": "5",
    }

    self.config["Notification"] = {"cooldown_hours": "1", "duration_seconds": "10"}

    self.config["Excluded"] = {
      "processes": default_excluded,
      "sensitive_keywords": "password,bank,wallet,login,auth",
    }

    self.config["Database"] = {"path": "data/app_patterns.db"}

    self.config["Logging"] = {"level": "INFO", "file": "logs/app.log"}

    # Write config to file
    with open(self.config_path, "w", encoding="utf-8") as f:
      self.config.write(f)

  def get_str(self, section: str, key: str, default: str = "") -> str:
    """Get string value from config.

    Args:
      section: Configuration section name
      key: Configuration key name
      default: Default value if key not found

    Returns:
      Configuration value or default
    """
    return self.config.get(section, key, fallback=default)

  def get_int(self, section: str, key: str, default: int = 0) -> int:
    """Get integer value from config.

    Args:
      section: Configuration section name
      key: Configuration key name
      default: Default value if key not found

    Returns:
      Configuration value or default
    """
    return self.config.getint(section, key, fallback=default)

  def get_float(self, section: str, key: str, default: float = 0.0) -> float:
    """Get float value from config.

    Args:
      section: Configuration section name
      key: Configuration key name
      default: Default value if key not found

    Returns:
      Configuration value or default
    """
    return self.config.getfloat(section, key, fallback=default)

  def get_bool(self, section: str, key: str, default: bool = False) -> bool:
    """Get boolean value from config.

    Args:
      section: Configuration section name
      key: Configuration key name
      default: Default value if key not found

    Returns:
      Configuration value or default
    """
    return self.config.getboolean(section, key, fallback=default)

  def get_excluded_processes(self) -> Set[str]:
    """Get set of excluded process names.

    Returns:
      Set of process names to exclude from monitoring
    """
    processes_str: str = self.get_str("Excluded", "processes")
    return {proc.strip() for proc in processes_str.split(",") if proc.strip()}

  def get_sensitive_keywords(self) -> Set[str]:
    """Get set of sensitive keywords to filter out.

    Returns:
      Set of keywords that indicate sensitive apps
    """
    keywords_str: str = self.get_str("Excluded", "sensitive_keywords")
    return {keyword.strip().lower() for keyword in keywords_str.split(",") if keyword.strip()}

  def get_database_path(self) -> Path:
    """Get database file path.

    Returns:
      Path to SQLite database file
    """
    db_path_str: str = self.get_str("Database", "path")
    db_path: Path = Path(db_path_str)

    # If relative path, make it relative to config directory
    if not db_path.is_absolute():
      db_path = self.config_path.parent / db_path

    return db_path

  def get_log_path(self) -> Path:
    """Get log file path.

    Returns:
      Path to log file
    """
    log_path_str: str = self.get_str("Logging", "file")
    log_path: Path = Path(log_path_str)

    # If relative path, make it relative to config directory
    if not log_path.is_absolute():
      log_path = self.config_path.parent / log_path

    return log_path

  def get_config_dict(self) -> ConfigDict:
    """Get configuration as a dictionary.

    Returns:
      Configuration dictionary
    """
    return ConfigDict(
      min_occurrences=self.get_int("Pattern", "min_occurrences", 3),
      min_confidence=self.get_float("Pattern", "min_confidence", 60.0),
      session_timeout_minutes=self.get_int("Pattern", "session_timeout_minutes", 5),
      notification_cooldown_hours=self.get_int("Notification", "cooldown_hours", 1),
      excluded_processes=self.get_excluded_processes(),
    )

  def save(self) -> None:
    """Save current configuration to file."""
    with open(self.config_path, "w", encoding="utf-8") as f:
      self.config.write(f)
    logging.info(f"Saved configuration to {self.config_path}")
