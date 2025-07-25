# YouTube Request Interceptor

A Chrome extension that intercepts and monitors HTTP requests made to YouTube, with the ability to modify responses using custom functions, fetch from an external server, or detect and mark bot comments.

## Features

- Intercepts all XMLHttpRequests made to YouTube domains
- Displays a draggable panel on YouTube pages showing intercepted requests
- Shows request method, URL path, timestamp, and status codes
- Color-coded status codes (green for success, red for errors)
- Stores up to 100 recent requests
- Click on a request to see detailed information
- Modify responses using custom JavaScript functions
- Dynamically fetch modified responses from an external server
- Detect and mark bot comments using an AI-powered API
- **NEW:** Right-click on comments to report them as bots

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

## Bot Detection

The extension can detect and mark bot comments in YouTube videos:

1. Click the "Bot Detection" button in the panel
2. Check "Enable Bot Detection" to activate the feature
3. Enter your API key for the bot detection service
4. Enter the API server URL
5. Click "Test API Connection" to verify the connection
6. Configure additional settings as needed:
   - Mark bot names with [BOT] tag
   - Enable right-click report menu
7. Click "Save Settings" to apply your changes

### Right-Click Report Menu

You can manually report comments as bots:

1. Enable the "Enable right-click report menu" option in Bot Detection settings
2. Right-click on any YouTube comment
3. Select "Report as Bot" from the context menu
4. The comment will be sent to the API for reporting
5. A "REPORTED" badge will be added to the comment

### Bot Detection API

The extension uses the following API endpoints:

- `/evaluate`: Evaluates comments to determine if they are from bots
  - Request: `{ evaluate: [{ author_name: string, content: string }], api_key: string }`
  - Response: `{ result: number[], is_bot: boolean[] }`

- `/report`: Reports a comment as a bot
  - Request: `{ author_name: string, content: string, is_bot: boolean, api_key: string }`
  - Response: `{ result: "success" }`

Example API implementation (Python with FastAPI):

```python
@app.post("/evaluate")
async def evaluate(item: EvaluateRequest) -> dict[str, Sequence[float | bool]]:
  # Validate request
  if not item.evaluate:
    raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Author name and content are required")
  
  # Validate API key
  try:
    _ = ph.verify(api_key_hashed, item.api_key)
  except VerifyMismatchError as e:
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid API key") from e
  
  # Extract data
  author_names = [i.author_name for i in item.evaluate]
  contents = [i.content for i in item.evaluate]
  
  # Evaluate using your bot detection model
  result: list[float] = model.evaluate(author_names, contents)
  is_bot = [result > 0.9 for result in result]
  
  return {"result": result, "is_bot": is_bot}

@app.post("/report")
async def report(item: ReportRequest):
  # Validate API key
  try:
    _ = ph.verify(api_key_hashed, item.api_key)
  except VerifyMismatchError as e:
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid API key") from e
  
  if not item.author_name or not item.content:
    raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Author name and content are required")
  
  # Store in database
  entry_id = str(uuid.uuid4())
  kv_store[entry_id] = {
    "author_name": item.author_name,
    "content": item.content,
    "is_bot": item.is_bot,
    "timestamp": datetime.now().isoformat()
  }
  
  return {"result": "success"}
```

### Custom Function Example

```javascript
function(response, url) {
  // Example: Modify a YouTube API response based on URL
  if (url.includes('/youtubei/v1/browse')) {
    // Check if this is a comments response
    if (response && response.frameworkUpdates && 
        response.frameworkUpdates.entityBatchUpdate && 
        response.frameworkUpdates.entityBatchUpdate.mutations) {
      
      // Process comment mutations
      const mutations = response.frameworkUpdates.entityBatchUpdate.mutations;
      
      for (const mutation of mutations) {
        if (mutation.payload && mutation.payload.commentEntityPayload) {
          const payload = mutation.payload.commentEntityPayload;
          
          // Example: Mark comments with specific keywords as bots
          if (payload.author && payload.properties && payload.properties.content) {
            const content = payload.properties.content.content.toLowerCase();
            if (content.includes('subscribe to my channel') || 
                content.includes('check out my video')) {
              payload.author.displayName += ' [BOT]';
            }
          }
        }
      }
    }
  }
  return response;
}
```

**Note:** Due to Chrome's security restrictions, response modification capabilities are limited in Manifest V3. The extension will track which responses would be modified, but actual modification requires additional permissions or workarounds.

## Development

The extension consists of these main files:

- `manifest.json`: Extension configuration (Manifest V3 compliant)
- `background.js`: Background script that intercepts requests and processes responses
- `content.js`: Content script that displays the UI on YouTube pages
- `rules.json`: Declarative rules for network request handling

## Permissions

This extension requires the following permissions:
- `webRequest`: To intercept web requests (non-blocking)
- `storage`: To store intercepted requests and response modification settings
- `declarativeNetRequest`: For limited response modification capabilities
- `contextMenus`: For the right-click report menu
- Host permissions for YouTube domains and external servers 