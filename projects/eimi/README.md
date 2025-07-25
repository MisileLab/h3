# YouTube Bot Detector

A Chrome extension that detects and marks bot comments on YouTube videos with AI-powered analysis. This production-ready extension helps you identify potential spam and bot comments in YouTube comment sections.

## Features

- Detects bot comments on YouTube videos using AI-powered analysis
- Marks bot authors with [BOT] tag in comments
- Right-click on suspicious comments to report them as bots
- Configurable API connection for bot detection service
- Secure design with predefined modification functions
- Minimal permissions required for YouTube domains only
- No icons or unnecessary files for lightweight installation

## Installation

1. Download the latest release from the releases section
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" in the top right
4. Click "Load unpacked" and select the extracted folder
5. The extension is now installed and will automatically run on YouTube pages

## Usage

1. Go to any YouTube page with comments (e.g., a video page)
2. A small panel will appear in the bottom right corner
3. Click "Bot Detection" to configure the bot detection settings
4. Enter your API key and server URL for the bot detection service
5. Enable the bot detection feature
6. Bot comments will be automatically marked with [BOT] tag
7. Right-click on any suspicious comment to report it as a bot

## Response Modification

The extension includes predefined functions for modifying YouTube responses:

1. Click the "Modify Responses" button in the panel
2. Check "Enable Response Modification" to activate response modification
3. Select a predefined function from the dropdown menu:
   - `markBotComments`: Adds [BOT] tag to comments matching spam patterns
   - `removeSpamLinks`: Removes URLs from comments to prevent spam
4. Enter your API server URL (required for bot detection)
5. Enter your API key for authentication
6. Click "Save Settings" to apply your changes

### Bot Detection API

The extension uses the following API endpoints:

- `/evaluate`: Evaluates comments to determine if they are from bots
  - Request: `{ evaluate: [{ author_name: string, content: string }], api_key: string }`
  - Response: `{ result: number[], is_bot: boolean[] }`

- `/report`: Reports a comment as a bot
  - Request: `{ author_name: string, content: string, is_bot: boolean, api_key: string }`
  - Response: `{ result: "success" }`

## Security Features

This extension follows best practices for security:

- Uses predefined functions instead of eval() or Function constructor
- Implements strict Content Security Policy
- Requests minimal permissions (only YouTube domains)
- Follows Manifest V3 requirements for Chrome extensions
- No external dependencies or third-party libraries

## Development

The extension consists of these main files:

- `manifest.json`: Extension configuration (Manifest V3 compliant)
- `background.js`: Background script that intercepts requests and processes responses
- `content.js`: Content script that displays the UI on YouTube pages
- `rules.json`: Declarative rules for network request handling

To build the extension for distribution:

```
# PowerShell
.\zip.ps1
```

This will create a versioned zip file ready for distribution.

## Permissions

This extension requires the following permissions:
- `webRequest`: To intercept web requests (non-blocking)
- `storage`: To store settings and configuration
- `declarativeNetRequest`: For network request handling
- `contextMenus`: For the right-click report menu
- Host permissions for YouTube domains only 