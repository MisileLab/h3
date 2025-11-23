# ArcX Overwolf App

In-game overlay for ArcX EV prediction system.

## Setup

1. Install pnpm (if not already installed):
   ```bash
   # Windows
   iwr https://get.pnpm.io/install.ps1 -useb | iex

   # Linux/macOS
   curl -fsSL https://get.pnpm.io/install.sh | sh -
   ```

2. Install dependencies:
   ```bash
   pnpm install
   ```

3. Build TypeScript:
   ```bash
   pnpm run build
   ```

3. Load in Overwolf:
   - Open Overwolf
   - Settings → Support → Development options
   - Load unpacked extension
   - Select this `overwolf/` directory

## Development

Run in watch mode:
```bash
pnpm run watch
```

Update dependencies:
```bash
pnpm update
```

## Structure

- `manifest.json` - Overwolf app configuration
- `src/index.html` - Main overlay UI
- `src/overlay.ts` - Overlay logic
- `src/api-client.ts` - Backend API client
- `src/background.ts` - Background script

## Features

- Real-time EV display
- Stay vs Extract recommendation
- Risk profile adjustment
- User feedback buttons
- Run start/end tracking

## Backend Connection

The overlay connects to the Python backend at `http://127.0.0.1:8765`.

Make sure the backend is running before starting the game:
```bash
cd ../backend
python serve.py
```

## Usage

1. Start the backend server
2. Launch the game
3. The overlay will appear automatically
4. Click "시작" to start a run
5. EV values will update every 0.5 seconds
6. Use feedback buttons to rate recommendations
7. Click "종료" when extracting

## Troubleshooting

### Overlay not appearing
- Check if Overwolf is running
- Verify the game is supported (game_id in manifest.json)
- Check Overwolf developer console for errors

### Backend connection failed
- Ensure backend server is running on port 8765
- Check firewall settings
- Look at browser console (F12) for errors

### No EV predictions
- Backend may be loading the model
- Buffer might be filling (needs 32 frames)
- Check backend logs for errors
