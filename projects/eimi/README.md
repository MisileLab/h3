# YouTube Request Interceptor

A Chrome extension that intercepts and monitors HTTP requests made to YouTube, with the ability to modify responses using custom functions or by fetching from an external server.

## Features

- Intercepts all XMLHttpRequests made to YouTube domains
- Displays a draggable panel on YouTube pages showing intercepted requests
- Shows request method, URL path, timestamp, and status codes
- Color-coded status codes (green for success, red for errors)
- Stores up to 100 recent requests
- Click on a request to see detailed information
- **NEW:** Modify responses using custom JavaScript functions
- **NEW:** Dynamically fetch modified responses from an external server

## Installation

1. Clone or download this repository
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" in the top right
4. Click "Load unpacked" and select the `src` folder from this repository
5. The extension is now installed and will automatically run on YouTube pages

## Usage

1. Go to any YouTube page (e.g., youtube.com)
2. A small panel will appear in the bottom right corner
3. Click "Show Requests" to see the intercepted requests
4. The list will update automatically every 5 seconds
5. Click on any request to see detailed information including:
   - Full URL
   - Request method
   - Request type
   - Timestamp
   - Status code (if available)
   - Response time (if available)
6. You can drag the panel by its header to reposition it

## Response Modification

1. Click the "Modify Responses" button in the panel
2. Check "Enable Response Modification" to activate response modification
3. Enter your example server URL (optional)
4. Write a JavaScript function in the textarea that takes a response object and URL parameter
5. Click "Save Function" to apply your changes
6. Modified responses will be marked with a "MODIFIED" badge in the requests list

### Dynamic Response Modification with External Server

You can configure an external server to provide modified responses:

1. Enter the URL of your example server in the "Example Server URL" field
2. Click "Test" to verify the connection
3. The extension will send requests to this server with the original YouTube URL as a parameter
4. Your server should return a modified JSON response

Example server implementation (Node.js):

```javascript
const express = require('express');
const fetch = require('node-fetch');
const app = express();
const port = 3000;

app.get('/', async (req, res) => {
  const originalUrl = req.query.originalUrl;
  
  if (!originalUrl) {
    return res.status(400).json({ error: 'Missing originalUrl parameter' });
  }
  
  try {
    // Fetch the original response
    const originalResponse = await fetch(originalUrl);
    const originalData = await originalResponse.json();
    
    // Modify the response data
    const modifiedData = {
      ...originalData,
      modified: true,
      timestamp: new Date().toISOString()
    };
    
    // Return the modified data
    res.json(modifiedData);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`Example server listening at http://localhost:${port}`);
});
```

### Custom Function Example

```javascript
function(response, url) {
  // Example: Modify a YouTube API response based on URL
  if (url.includes('/youtubei/v1/browse')) {
    // Modify the data as needed
    if (response && response.contents) {
      response.contents.modified = true;
      response.contents.timestamp = new Date().toISOString();
    }
  }
  return response;
}
```

**Note:** Due to Chrome's security restrictions, response modification capabilities are limited in Manifest V3. The extension will track which responses would be modified, but actual modification requires additional permissions or workarounds.

## Development

The extension consists of these main files:

- `manifest.json`: Extension configuration (Manifest V3 compliant)
- `background.js`: Background script that intercepts requests
- `content.js`: Content script that displays the UI on YouTube pages
- `rules.json`: Declarative rules for network request handling

## Permissions

This extension requires the following permissions:
- `webRequest`: To intercept web requests (non-blocking)
- `storage`: To store intercepted requests and response modification settings
- `declarativeNetRequest`: For limited response modification capabilities
- Host permissions for YouTube domains and external servers 