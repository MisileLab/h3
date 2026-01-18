<!-- Parent: ../AGENTS.md -->

# EXTENSION SRC KNOWLEDGE BASE

## OVERVIEW
TypeScript sources for the MV3 extension: service worker orchestration, audio pipeline, UI, and message types.

## WHERE TO LOOK
| Task | Location | Notes |
| --- | --- | --- |
| Service worker | extension/src/sw.ts | WebSocket + state + routing |
| Content script | extension/src/content.ts | Caption overlay DOM |
| Offscreen audio | extension/src/offscreen.ts | AudioContext + worklet bridge |
| Worklet | extension/src/worklet-processor.ts | Resample + PCM16 frames |
| Popup logic | extension/src/popup.ts | Settings + start/stop |
| Types | extension/src/types.ts | Message contracts |

## CONVENTIONS
- types.ts is the single source of truth for extension message types.
- Worklet runs in isolated context, no imports.
- Audio frames are 480 samples at 24kHz.

## ANTI-PATTERNS
- Do not access DOM from service worker/offscreen.
- Do not change protocol message shapes without server updates.
