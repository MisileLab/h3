"""Device backend detection and management"""

from .backend import DeviceBackend, DeviceManager, detect_backend, device_manager

__all__ = ["DeviceBackend", "DeviceManager", "detect_backend", "device_manager"]
