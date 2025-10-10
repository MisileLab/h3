"""
Main entry point for TorchForge application.
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from .ui.main_window import MainWindow


def main() -> int:
  """Main entry point for the application."""
  app = QApplication(sys.argv)
  app.setApplicationName("TorchForge")
  app.setApplicationVersion("0.1.0")
  app.setOrganizationName("TorchForge Team")
  
  # Enable high DPI scaling
  app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
  app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
  
  window = MainWindow()
  window.show()
  
  return app.exec()


def run() -> None:
  """Convenient function to run the application."""
  sys.exit(main())


if __name__ == "__main__":
  run()