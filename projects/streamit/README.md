# StreamIt - Real-time Caption Overlay

Browser extension + proxy server that provides real-time live captions for Twitch/YouTube using OpenAI Realtime Transcription.

## Architecture

- **Chrome Extension (MV3)**: Captures audio from Twitch/YouTube, sends to server
- **FastAPI Server**: WebSocket proxy that forwards audio to OpenAI Realtime API
- **OpenAI Realtime API**: Provides real-time transcription
- **Security**: API keys stored on server only, never exposed to browser

## Features

- Real-time captions overlay on Twitch/YouTube live streams
- Multi-language support (auto, Korean, English)
- Customizable caption styling (font size, transparency, position)
- Usage-based metering (seconds-based billing/limits)
- Reconnection with exponential backoff
- Connection status indicator

## Requirements

- **Chrome**: 116+ (for MV3 and AudioWorklet support)
- **Python**: 3.11+
- **Docker**: (optional, for production deployment)

## Quick Start

### 1. Server Setup

```bash
# Clone repository
git clone <repo-url>
cd streamit

# Copy environment file
cp .env.example .env

# Edit .env and add your OpenAI API Key
OPENAI_API_KEY=sk-...
VIEWER_TOKENS=your-secure-token-here

# Install dependencies (or use Docker)
cd server
pip install -r requirements.txt

# Run server (dev)
./scripts/run_dev.sh

# Or run with Docker Compose
docker-compose up
```

### 2. Install Chrome Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `extension/` directory from this repository

### 3. Use Extension

1. Open a Twitch or YouTube live stream
2. Click extension icon
3. Configure:
   - Server URL: `ws://localhost:8000/ws/viewer` (dev) or `wss://yourdomain.com/ws/viewer` (prod)
   - User Token: Use the token from VIEWER_TOKENS in .env
   - Language: auto / ko / en
4. Click "Start Captions"
5. Captions should appear overlay within 5 seconds

## Server Configuration

### Environment Variables (.env)

```env
# OpenAI API
OPENAI_API_KEY=sk-...

# Viewer authentication (comma-separated)
VIEWER_TOKENS=token1,token2

# Server settings
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Metering limits (optional)
MAX_SECONDS_PER_USER=3600  # 1 hour per user per day
```

### Docker Deployment

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f server

# Stop
docker-compose down
```

### Nginx Reverse Proxy (WSS)

Example `nginx.conf` for production:

```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # WebSocket upgrade
    location /ws/ {
        proxy_pass http://server:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }

    # Health endpoint
    location /health {
        proxy_pass http://server:8000;
    }
}
```

## Testing

### Manual Testing

1. Start server: `cd server && ./scripts/run_dev.sh`
2. Load extension in Chrome
3. Open https://www.youtube.com/watch?v=<live-stream-id>
4. Start captions via extension popup
5. Verify:
   - Captions appear within 5 seconds
   - Stop button works
   - Connection status updates

### Automated Tests

```bash
cd server
pytest tests/
```

## Troubleshooting

### Tab Capture Permission Denied

- Ensure you're on HTTPS or localhost
- Chrome may block tab capture on certain sites
- Check extension permissions in chrome://extensions/

### No Audio / Silent Captions

- Verify the video has audio (not muted)
- Check browser console for errors
- Ensure AudioWorklet is loaded (offscreen.js logs)
- Try refreshing the page

### WebSocket Connection Failed

- Check server is running: `curl http://localhost:8000/health`
- Verify viewer token matches VIEWER_TOKENS in .env
- Check firewall/proxy settings
- Mixed content: HTTPS page cannot connect to WS://, use WSS://

### High Latency / Frequent Reconnects

- Check server logs for OpenAI errors
- Verify network stability
- Increase ping intervals in extension code
- Check OpenAI API quota/limits

### Captions Not Appearing

- Check content script is injected (DevTools → Console)
- Verify overlay CSS has z-index > video player
- Check for CSS conflicts (use Shadow DOM)
- Look for JS errors in DevTools console

## Security

- API keys are server-side only (never in extension)
- Bearer token authentication for viewers
- Metering to prevent abuse
- No sensitive data logged (tokens hashed)
- WSS (TLS) required for production

## Development

### Server

```bash
cd server
python -m pytest  # Run tests
./scripts/run_dev.sh  # Hot reload with uvicorn
```

### Extension

1. Make changes to extension files
2. Reload extension in `chrome://extensions/` (refresh button)
3. Reload the page being tested

### Project Structure

```
streamit/
├── README.md
├── .gitignore
├── .env.example
├── docker-compose.yml
├── server/                    # FastAPI proxy
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py            # FastAPI + WS endpoint
│   │   ├── config.py          # Settings
│   │   ├── auth.py            # Token validation
│   │   ├── meter.py           # Seconds metering
│   │   ├── openai_realtime.py # OpenAI WS client
│   │   ├── protocol.py        # Message schemas
│   │   └── logging_setup.py
│   ├── tests/
│   │   ├── test_meter.py
│   │   └── test_protocol.py
│   ├── scripts/
│   │   ├── run_dev.sh
│   │   └── run_prod.sh
│   └── nginx/
│       └── nginx.conf.example
└── extension/                 # Chrome MV3
    ├── manifest.json
    ├── sw.js                 # Service worker
    ├── offscreen.html
    ├── offscreen.js          # Audio processing
    ├── worklet-processor.js  # AudioWorklet
    ├── content.js            # Overlay UI
    ├── overlay.css
    ├── popup.html
    ├── popup.js
    └── icons/
```

## License

MIT
