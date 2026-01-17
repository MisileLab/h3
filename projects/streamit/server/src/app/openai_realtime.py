import base64
import json
import logging
from typing import Callable, Awaitable

import websockets
from websockets.asyncio.client import ClientConnection

from app.config import settings

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str], Awaitable[None]]
ErrorCallback = Callable[[str], Awaitable[None]]


class OpenAIRealtimeClient:
    """WebSocket client for OpenAI Realtime API."""

    def __init__(
        self,
        on_transcript_delta: TranscriptCallback,
        on_transcript_completed: TranscriptCallback,
        on_error: ErrorCallback,
        model: str = "gpt-realtime",
    ):
        self.on_transcript_delta = on_transcript_delta
        self.on_transcript_completed = on_transcript_completed
        self.on_error = on_error
        self.model = model
        self.ws: ClientConnection | None = None
        self._is_connected = False

    async def connect(self, language: str = "auto"):
        url = f"{settings.openai_ws_url}?model={self.model}"
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"}

        logger.info(f"Connecting to OpenAI Realtime: {url}")

        try:
            self.ws = await websockets.connect(url, extra_headers=headers)
            self._is_connected = True
            logger.info("OpenAI Realtime connected")

            await self._configure_session(language)
            return True

        except Exception as e:
            logger.error(f"OpenAI Realtime connection failed: {e}")
            await self._safe_callback(self.on_error, str(e))
            return False

    async def _configure_session(self, language: str):
        """Send session.update to configure transcription-only mode."""
        lang_code = None if language == "auto" else language

        config = {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {
                            "model": "gpt-4o-transcribe",
                            "language": lang_code,
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "silence_duration_ms": 500,
                        },
                    }
                },
                "include": ["item.input_audio_transcription.logprobs"],
            },
        }

        await self._send_json(config)
        logger.info(f"Session configured for language: {language}")

    async def send_audio(self, pcm16_bytes: bytes):
        """Send PCM16 audio bytes to OpenAI."""
        if not self._is_connected or self.ws is None:
            return

        base64_audio = base64.b64encode(pcm16_bytes).decode("utf-8")
        message = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio,
        }

        await self._send_json(message)

    async def _send_json(self, data: dict):
        """Send JSON message to WebSocket."""
        try:
            if self.ws:
                await self.ws.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to send to OpenAI: {e}")
            await self._safe_callback(self.on_error, f"Send failed: {e}")

    async def listen(self):
        """Listen for incoming messages from OpenAI."""
        if not self.ws:
            return

        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from OpenAI: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("OpenAI WebSocket closed")
            self._is_connected = False
            await self._safe_callback(self.on_error, "OpenAI connection closed")

        except Exception as e:
            logger.error(f"OpenAI listener error: {e}")
            self._is_connected = False
            await self._safe_callback(self.on_error, f"Listener error: {e}")

    async def _handle_message(self, data: dict):
        """Handle incoming message from OpenAI."""
        msg_type = data.get("type")

        if msg_type == "conversation.item.input_audio_transcription.delta":
            if delta := data.get("delta", ""):
                await self._safe_callback(self.on_transcript_delta, delta)

        elif msg_type == "conversation.item.input_audio_transcription.completed":
            if transcript := data.get("transcript", ""):
                await self._safe_callback(self.on_transcript_completed, transcript)

        elif msg_type == "error":
            error_msg = data.get("error", {}).get("message", "Unknown error")
            logger.error(f"OpenAI error: {error_msg}")
            await self._safe_callback(self.on_error, f"OpenAI: {error_msg}")

        elif msg_type == "session.updated":
            logger.info("OpenAI session updated")

    async def _safe_callback(
        self, callback: TranscriptCallback | ErrorCallback, *args: str
    ) -> None:
        """Safely call async callback."""
        try:
            await callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    async def close(self):
        """Close WebSocket connection."""
        self._is_connected = False
        if self.ws:
            try:
                await self.ws.close()
                logger.info("OpenAI Realtime disconnected")
            except Exception as e:
                logger.error(f"Error closing OpenAI WS: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if connected to OpenAI."""
        return self._is_connected
