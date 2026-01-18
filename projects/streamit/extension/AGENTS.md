<!-- Parent: ../AGENTS.md -->

# EXTENSION KNOWLEDGE BASE

## OVERVIEW
Chrome MV3 extension that captures tab audio, streams PCM16 to the server, and renders live captions on Twitch/YouTube.

## STRUCTURE
```
extension/
├── src/               # TypeScript sources
├── dist/              # esbuild output
├── manifest.json      # MV3 config + entrypoints
├── popup.html         # Popup UI shell
├── offscreen.html     # Offscreen document shell
└── overlay.css        # Caption overlay styling
```

## WHERE TO LOOK
| Task | Location | Notes |
| --- | --- | --- |
| Orchestration | extension/src/sw.ts | WebSocket, tab capture, message routing |
| Caption UI | extension/src/content.ts | Overlay rendering + settings |
| Audio capture | extension/src/offscreen.ts | AudioContext + AudioWorklet wiring |
| Audio processing | extension/src/worklet-processor.ts | Resample 48kHz→24kHz, PCM16 |
| Popup logic | extension/src/popup.ts | Settings + start/stop |
| Message types | extension/src/types.ts | Component and server protocol types |

## CONVENTIONS
- MV3 offscreen document required for AudioWorklet.
- WebSocket is binary for audio frames, JSON for control messages.
- Settings flow: popup -> service worker -> offscreen + content.

## ANTI-PATTERNS
- Do not put API keys or secrets in extension code.
- Do not change message types without updating server protocol.

## NOTES
- Content script must keep overlay z-index above video players.
- Worklet is built as IIFE and exposed via web_accessible_resources.
