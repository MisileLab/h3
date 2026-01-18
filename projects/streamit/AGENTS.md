# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-18
**Commit:** ba629fc94
**Branch:** main

## OVERVIEW
StreamIt is a Chrome MV3 extension plus a FastAPI server that proxies audio to OpenAI Realtime for live captions on Twitch/YouTube.

## STRUCTURE
```
streamit/
├── extension/        # Chrome MV3 extension (TypeScript)
├── server/           # FastAPI WebSocket proxy (Python)
├── docker-compose.yml
├── .env.example
└── README.md
```

## WHERE TO LOOK
| Task | Location | Notes |
| --- | --- | --- |
| Extension service worker | extension/src/sw.ts | Orchestrates WebSocket, audio, messaging |
| Caption overlay | extension/src/content.ts | DOM overlay rendering |
| Audio pipeline | extension/src/offscreen.ts, extension/src/worklet-processor.ts | Tab capture + AudioWorklet |
| Server WebSocket | server/src/app/main.py | /ws/viewer handler |
| OpenAI integration | server/src/app/openai_realtime.py | Realtime + translation |
| Protocol schema | extension/src/types.ts, server/src/app/protocol.py | Must stay in sync |
| Usage metering | server/src/app/meter.py | Seconds-based limits |

## CODE MAP
| Symbol | Type | Location | Refs | Role |
| --- | --- | --- | --- | --- |
| websocket_viewer | function | server/src/app/main.py | WebSocket entrypoint | Orchestrates session |
| OpenAIRealtimeClient | class | server/src/app/openai_realtime.py | OpenAI client | Realtime transcription |
| StreamItAudioProcessor | class | extension/src/worklet-processor.ts | AudioWorklet | Resample + PCM16 |

## CONVENTIONS
- Manual monorepo: no root workspace tooling; extension and server are separate.
- TypeScript runs in strict mode; no lint/format configs present.
- Python uses uv tooling; pyproject.toml is the source of truth.
- Protocol contract is duplicated in TypeScript and Python and must be updated together.

## ANTI-PATTERNS (THIS PROJECT)
- Never expose OpenAI API keys in the extension or browser context.
- Do not change protocol fields on one side without updating the other.

## UNIQUE STYLES
- Chrome MV3 with offscreen document + AudioWorklet.
- Server uses WebSocket proxy with cache-aside transcript storage.

## COMMANDS
```bash
# Extension
cd extension
yarn build
yarn watch
yarn typecheck

# Server
cd server
./scripts/run_dev.sh
./scripts/run_prod.sh
pytest tests/

# Docker
cd ..
docker-compose up
```

## NOTES
- Sample rate contract: extension resamples to 24kHz PCM16; server expects 24kHz.
- Usage metering uses bytes/48000.0 to derive seconds.
