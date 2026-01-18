<!-- Parent: ../../AGENTS.md -->

# SERVER APP KNOWLEDGE BASE

## OVERVIEW
FastAPI app modules implementing WebSocket handling, OpenAI realtime proxying, auth, metering, and caching.

## WHERE TO LOOK
| Task | Location | Notes |
| --- | --- | --- |
| WebSocket flow | server/src/app/main.py | JSON control + binary audio |
| OpenAI proxy | server/src/app/openai_realtime.py | Realtime events + translation |
| Protocol models | server/src/app/protocol.py | Pydantic message schemas |
| Auth | server/src/app/auth.py | Token validate + hash |
| Metering | server/src/app/meter.py | Per-token daily usage |
| Cache | server/src/app/audio_cache.py | SQLite cache-aside |
| Settings | server/src/app/config.py | Env-based settings |
| Logging | server/src/app/logging_setup.py | Root logger setup |

## CONVENTIONS
- main.py wires callbacks to OpenAIRealtimeClient.
- protocol.py must match extension/src/types.ts.
- Audio cache keyed by (audio_hash, target_lang).

## ANTI-PATTERNS
- Do not log raw tokens.
- Do not bypass UsageMeter checks.
