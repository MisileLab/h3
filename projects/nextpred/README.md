# Next Action Predictor

A cross-platform OS-level action pattern learning and suggestion system that learns your app usage patterns and suggests next actions via notifications.

## Features

- **Cross-platform support**: Works on Windows, Linux, and macOS
- **Pattern learning**: Automatically detects repeated app execution sequences
- **Smart notifications**: Suggests next apps based on learned patterns
- **System tray interface**: Easy control via system tray menu
- **Privacy-focused**: All data stored locally, no cloud connectivity
- **Configurable**: Adjustable sensitivity and notification settings

## How It Works

1. **Window Monitoring**: Continuously monitors all visible windows and mouse position
2. **Focus Tracking**: Tracks which window has focus and mouse cursor position
3. **Pattern Detection**: Identifies repeated sequences based on window focus and mouse interactions
4. **Context-Aware Learning**: Uses both current window and mouse position to predict next actions
5. **Confidence Scoring**: Calculates confidence based on pattern frequency and context
6. **Smart Suggestions**: Shows notifications when a pattern is likely to continue
7. **User Feedback**: Learns from user interactions to improve suggestions

## Installation

### Prerequisites

- Python 3.10 or higher
- `uv` package manager (recommended)
- **Administrator privileges** (required for Windows API access to monitor all windows)

### Quick Install with uv

```bash
# Clone the repository
git clone <repository-url>
cd next-action-predictor

# Install dependencies with uv
uv sync

# Run the application (requires administrator privileges)
# On Windows: Right-click Command Prompt/PowerShell and "Run as administrator"
uv run python src/main.py
```

### Alternative Installation with pip

```bash
# Clone the repository
git clone <repository-url>
cd next-action-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install psutil plyer pystray Pillow

# Run the application (requires administrator privileges)
# On Windows: Right-click Command Prompt/PowerShell and "Run as administrator"
python src/main.py
```

## Usage

### Starting the Application

```bash
# Using uv (requires administrator privileges)
uv run python src/main.py

# Using pip (after activating venv, requires administrator privileges)
python src/main.py
```

**Important**: This application requires administrator privileges to:
- Access Windows API for monitoring all visible windows
- Track mouse position and window focus accurately
- Provide comprehensive pattern learning

The application will start in the background and add an icon to your system tray.

### System Tray Menu

Right-click the system tray icon to access:

- **Learning Status**: View current learning progress
- **View Statistics**: See detailed usage statistics
- **Recent Patterns**: View learned patterns
- **Test Notification**: Test the notification system
- **Analyze Now**: Manually trigger pattern analysis
- **Clear Cooldowns**: Reset notification cooldowns
- **Settings**: View current configuration
- **Help**: Get usage information
- **Exit**: Close the application

### Configuration

The application creates a configuration file at:

- **Windows**: `%APPDATA%\NextActionPredictor\config.ini`
- **Linux/macOS**: `~/.config/next-action-predictor/config.ini`

Default settings:

```ini
[Pattern]
min_occurrences = 3
min_confidence = 60.0
session_timeout_minutes = 5

[Notification]
cooldown_hours = 1
duration_seconds = 10

[Excluded]
processes = svchost.exe,System,dwm.exe,explorer.exe
sensitive_keywords = password,bank,wallet,login,auth
```

## Development

### Project Structure

```
next-action-predictor/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config.py               # Configuration management
│   ├── database.py             # SQLite database interface
│   ├── monitor.py              # Process monitoring
│   ├── pattern_engine.py       # Pattern recognition
│   ├── notifier.py             # Notifications
│   ├── tray_app.py             # System tray interface
│   └── typedefs.py             # Type definitions
├── tests/
│   ├── __init__.py
│   ├── test_database.py
│   └── test_pattern_engine.py
├── resources/                  # Icons and assets
├── data/                       # Database storage (created at runtime)
├── logs/                       # Log files (created at runtime)
├── pyproject.toml              # Project configuration
├── uv.lock                     # Dependency lock file
└── README.md
```

### Running Tests

```bash
# Using uv
uv run pytest

# Using pip
pytest
```

### Code Quality

```bash
# Type checking
uv run mypy src/

# Linting
uv run ruff check src/

# Formatting
uv run ruff format src/
```

## How Patterns Are Learned

### Pattern Detection Rules

- **Minimum occurrences**: Pattern must appear at least 3 times (configurable)
- **Confidence threshold**: Only suggest patterns with ≥60% confidence (configurable)
- **Sequence length**: Detects patterns of 2-5 apps
- **Session grouping**: Apps executed within 5 minutes are grouped as one session

### Confidence Calculation

```
Confidence = (Pattern Occurrences / Total Attempts) × 100
```

Example:
- Pattern occurs 4 times out of 5 attempts = 80% confidence
- Pattern occurs 2 times out of 4 attempts = 50% confidence (below threshold)

### Example Pattern Learning

```
Day 1: Chrome → Notion → VS Code
Day 2: Chrome → Notion → VS Code  
Day 3: Chrome → Notion → VS Code

Result: Pattern ["chrome.exe", "Notion.exe", "Code.exe"] with 100% confidence

Day 4: User runs Chrome → Notion
System shows notification: "Would you like to run VS Code?"
```

## Privacy and Security

- **Local storage only**: All data stored in local SQLite database
- **No network access**: Application never connects to the internet
- **Sensitive app filtering**: Automatically excludes apps with sensitive keywords
- **User control**: Full control over data deletion and settings

### Data Collected

- Application name (process name only)
- Process ID
- Window title (for context)
- Mouse position (x, y coordinates)
- Current focused window
- Start timestamp
- Session grouping information

### Data NOT Collected

- Window content or screenshots
- File names or paths
- User input or keystrokes
- Network activity
- Personal information
- Click tracking (only position, not clicks)

## Troubleshooting

### Common Issues

**Application doesn't start**
- Check Python version (requires 3.10+)
- Ensure all dependencies are installed
- **Run with administrator privileges** (required on Windows)
- Check log files in `logs/app.log`

**"Access denied" errors**
- Ensure running with administrator privileges
- On Windows: Right-click terminal and "Run as administrator"
- Check if antivirus is blocking the application

**Mouse position not tracking**
- Verify administrator privileges
- Check if security software is blocking API access
- Restart application with elevated privileges

**No notifications appearing**
- Test notification system from tray menu
- Check system notification settings
- Verify notification cooldown period

**High CPU usage**
- Check excluded processes list
- Reduce polling frequency in settings
- Monitor system resource usage
- Ensure running with proper privileges (insufficient privileges can cause excessive polling)

**Patterns not being learned**
- Ensure minimum usage threshold is met
- Check session timeout settings
- Verify app monitoring is working
- **Run with administrator privileges** to ensure all windows are detected
- Check if mouse tracking is working (see logs)

### Log Files

Check `logs/app.log` for detailed information:

```bash
# View recent logs
tail -f logs/app.log

# Search for errors
grep "ERROR" logs/app.log
```

### Database Location

Database files are stored at:

- **Windows**: `%APPDATA%\NextActionPredictor\app_patterns.db`
- **Linux**: `~/.local/share/next-action-predictor/app_patterns.db`
- **macOS**: `~/Library/Application Support/NextActionPredictor/app_patterns.db`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions:

- Create an issue on GitHub
- Check the troubleshooting section
- Review log files for error details

## Changelog

### v0.1.0 (Initial Release)

- Cross-platform process monitoring
- Pattern detection and learning
- System tray interface
- Notification system
- SQLite database storage
- Configuration management
- Basic test suite