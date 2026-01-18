<!-- Parent: ../AGENTS.md -->

# SERVER KNOWLEDGE BASE

## OVERVIEW
FastAPI WebSocket proxy that authenticates viewers, meters usage, and forwards audio to OpenAI Realtime.

## STRUCTURE
```
server/
├── src/app/           # FastAPI app + core modules
├── tests/             # pytest unit tests
├── scripts/           # dev/prod launchers
├── Dockerfile         # uv-based container
└── pyproject.toml     # dependencies + build config
```

## WHERE TO LOOK
| Task | Location | Notes |
| --- | --- | --- |
| WebSocket handler | server/src/app/main.py | /ws/viewer orchestration |
| OpenAI client | server/src/app/openai_realtime.py | Realtime + translation |
| Protocol models | server/src/app/protocol.py | Pydantic schema contract |
| Auth | server/src/app/auth.py | Token validation + hashing |
| Metering | server/src/app/meter.py | Seconds-based limits |
| Audio cache | server/src/app/audio_cache.py | SQLite cache-aside |
| Settings | server/src/app/config.py | pydantic-settings env loader |

## CONVENTIONS
- Protocol schema must stay in sync with extension/src/types.ts.
- Audio sample rate contract is 24kHz PCM16.
- Usage metering uses bytes/48000.0 to derive seconds.

## ANTI-PATTERNS
- Never log raw viewer tokens; use hash_token().
- Do not modify protocol fields without updating extension types.

## NOTES
- WebSocket accepts JSON control + binary audio frames.
- Audio cache keys are (audio_hash, target_lang).
