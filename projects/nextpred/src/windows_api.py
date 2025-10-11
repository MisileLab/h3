"""
Cross-platform window management and focus control.
Windows implementation based on StackOverflow solutions for reliable window activation.
"""

import platform
import psutil
import logging

# Platform detection
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"

def focus_process_by_name(process_name: str) -> bool:
    """
    Focus a window belonging to a process by name.
    
    Args:
        process_name: Name of the process (e.g., 'chrome.exe', 'notepad.exe')
        
    Returns:
        True if successful, False otherwise
    """
    if IS_WINDOWS:
        return _focus_process_windows(process_name)
    elif IS_LINUX:
        return _focus_process_linux(process_name)
    elif IS_MACOS:
        return _focus_process_macos(process_name)
    else:
        logging.warning(f"Window focus not implemented for {platform.system()}")
        return False

def _focus_process_windows(process_name: str) -> bool:
    """Windows-specific implementation using Windows APIs."""
    try:
        import ctypes
        from ctypes import wintypes
        
        # Windows API functions
        user32 = ctypes.windll.user32
        
        SetForegroundWindow = user32.SetForegroundWindow
        SetForegroundWindow.argtypes = [wintypes.HWND]
        SetForegroundWindow.restype = wintypes.BOOL
        
        ShowWindow = user32.ShowWindow
        ShowWindow.argtypes = [wintypes.HWND, wintypes.INT]
        ShowWindow.restype = wintypes.BOOL
        
        IsWindowVisible = user32.IsWindowVisible
        IsWindowVisible.argtypes = [wintypes.HWND]
        IsWindowVisible.restype = wintypes.BOOL
        
        GetWindowThreadProcessId = user32.GetWindowThreadProcessId
        GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        GetWindowThreadProcessId.restype = wintypes.DWORD
        
        # Find the process
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'].lower() == process_name.lower():
                # Try to find and focus the window
                try:
                    import win32gui
                    
                    def enum_windows_callback(hwnd, pid):
                        if GetWindowThreadProcessId(hwnd, None) == pid and IsWindowVisible(hwnd):
                            # Restore and focus the window
                            ShowWindow(hwnd, 9)  # SW_RESTORE
                            SetForegroundWindow(hwnd)
                            return True
                        return False
                    
                    win32gui.EnumWindows(enum_windows_callback, proc.info['pid'])
                    return True
                    
                except ImportError:
                    logging.warning("win32gui not available, using basic focus")
                    return False
                    
    except Exception as e:
        logging.error(f"Error focusing process on Windows: {e}")
        return False

def _focus_process_linux(process_name: str) -> bool:
    """Linux-specific implementation using wmctrl."""
    try:
        import subprocess
        # Try to find and focus the window using wmctrl
        result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if process_name.lower() in line.lower():
                    window_id = line.split()[0]
                    subprocess.run(['wmctrl', '-i', '-a', window_id])
                    return True
        return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Error focusing process on Linux: {e}")
        return False

def _focus_process_macos(process_name: str) -> bool:
    """macOS-specific implementation using AppleScript."""
    try:
        import subprocess
        # Use AppleScript to focus the application
        script = f'''
        tell application "System Events"
            set frontmost of first process whose name is "{process_name}" to true
        end tell
        '''
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        logging.error(f"Error focusing process on macOS: {e}")
        return False

def get_active_window_process() -> str:
    """
    Get the process name of the currently active window.
    
    Returns:
        Process name or empty string if no window is active
    """
    if IS_WINDOWS:
        return _get_active_window_windows()
    elif IS_LINUX:
        return _get_active_window_linux()
    elif IS_MACOS:
        return _get_active_window_macos()
    else:
        return ""

def _get_active_window_windows() -> str:
    """Get active window process on Windows."""
    try:
        import win32gui
        hwnd = win32gui.GetForegroundWindow()
        if hwnd:
            _, pid = win32gui.GetWindowThreadProcessId(hwnd)
            if pid:
                proc = psutil.Process(pid)
                return proc.name()
    except Exception as e:
        logging.error(f"Error getting active window on Windows: {e}")
    return ""

def _get_active_window_linux() -> str:
    """Get active window process on Linux."""
    try:
        import subprocess
        result = subprocess.run(['xdotool', 'getwindowfocus', 'getwindowpid'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pid = int(result.stdout.strip())
            proc = psutil.Process(pid)
            return proc.name()
    except Exception as e:
        logging.error(f"Error getting active window on Linux: {e}")
    return ""

def _get_active_window_macos() -> str:
    """Get active window process on macOS."""
    try:
        import subprocess
        script = '''
        tell application "System Events"
            name of first process whose frontmost is true
        end tell
        '''
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logging.error(f"Error getting active window on macOS: {e}")
    return ""