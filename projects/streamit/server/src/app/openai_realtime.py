import base64
import logging
from typing import Awaitable, Callable, Any

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str], Awaitable[None]]
TranslationCallback = Callable[[str], Awaitable[None]]
ErrorCallback = Callable[[str], Awaitable[None]]


class OpenAIRealtimeClient:
    """OpenAI Realtime client using the official SDK."""

    def __init__(
        self,
        on_transcript_delta: TranscriptCallback,
        on_transcript_completed: TranscriptCallback,
        on_translation: TranslationCallback | None,
        on_error: ErrorCallback,
        model: str = "gpt-realtime",
    ):
        self.on_transcript_delta = on_transcript_delta
        self.on_transcript_completed = on_transcript_completed
        self.on_translation = on_translation
        self.on_error = on_error
        self.model = model
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._connection_cm: Any | None = None
        self.conn: Any | None = None
        self._is_connected = False
        self.source_language: str | None = None
        self.target_language: str = "auto"
        self.current_audio_hash: str | None = None
        self.last_transcript_language: str | None = None
        self.last_transcript_text: str | None = None

    async def connect(
        self, language: str = "auto", target_language: str = "auto"
    ) -> bool:
        logger.info(f"Connecting to OpenAI Realtime: model={self.model}")

        try:
            self._connection_cm = self.client.realtime.connect(model=self.model)
            if not self._connection_cm:
                raise RuntimeError("Failed to initialize realtime connection")
            self.conn = await self._connection_cm.__aenter__()
            self._is_connected = True
            self.target_language = target_language
            logger.info("OpenAI Realtime connected")

            await self._configure_session(language)
            return True

        except Exception as e:
            logger.error(f"OpenAI Realtime connection failed: {e}")
            await self._safe_callback(self.on_error, str(e))
            return False

    async def _configure_session(self, language: str) -> None:
        """Configure transcription-only mode."""
        if not self.conn:
            return

        lang_code = None if language == "auto" else language
        transcription_config: dict[str, Any] = {
            "model": "gpt-4o-transcribe",
        }
        if lang_code:
            transcription_config["language"] = lang_code

        await self.conn.session.update(
            session={
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000,
                        },
                        "transcription": transcription_config,
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 500,
                            "create_response": False,
                            "interrupt_response": False,
                        },
                    }
                },
            }
        )

        logger.info(
            "Session configured for language: %s (target=%s)",
            language,
            self.target_language,
        )

    async def send_audio(self, pcm16_bytes: bytes) -> None:
        """Send PCM16 audio bytes to OpenAI."""
        if not self._is_connected or not self.conn:
            return

        base64_audio = base64.b64encode(pcm16_bytes).decode("utf-8")
        try:
            await self.conn.input_audio_buffer.append(audio=base64_audio)
        except Exception as e:
            logger.error(f"Failed to send to OpenAI: {e}")
            await self._safe_callback(self.on_error, f"Send failed: {e}")

    async def listen(self) -> None:
        """Listen for incoming events from OpenAI."""
        if not self.conn:
            return

        try:
            async for event in self.conn:
                await self._handle_event(event)
        except Exception as e:
            logger.error(f"OpenAI listener error: {e}")
            self._is_connected = False
            await self._safe_callback(self.on_error, f"Listener error: {e}")

    async def _handle_event(self, event: Any) -> None:
        """Handle incoming event from OpenAI."""
        msg_type = self._get_value(event, "type")

        if msg_type == "conversation.item.input_audio_transcription.delta":
            delta = self._get_value(event, "delta")
            if delta:
                await self._safe_callback(self.on_transcript_delta, delta)

        elif msg_type == "conversation.item.input_audio_transcription.completed":
            transcript = self._get_value(event, "transcript")
            if transcript:
                language = self._get_value(event, "language")
                if language:
                    self.source_language = language
                self.last_transcript_language = self.source_language
                self.last_transcript_text = transcript

                if self._should_translate():
                    translated = await self._translate_if_needed(transcript)
                    if translated is not None and self.on_translation:
                        await self._safe_callback(self.on_translation, translated)
                    else:
                        await self._safe_callback(
                            self.on_transcript_completed, transcript
                        )
                else:
                    await self._safe_callback(self.on_transcript_completed, transcript)

        elif msg_type == "error":
            error_obj = self._get_value(event, "error")
            error_msg = self._get_value(error_obj, "message") if error_obj else None
            if not error_msg:
                error_msg = "Unknown error"
            logger.error(f"OpenAI error: {error_msg}")
            await self._safe_callback(self.on_error, f"OpenAI: {error_msg}")

        elif msg_type == "session.updated":
            logger.info("OpenAI session updated")

    @property
    def current_target_language(self) -> str:
        return self.target_language

    async def _safe_callback(
        self, callback: TranscriptCallback | ErrorCallback, *args: str
    ) -> None:
        """Safely call async callback."""
        try:
            await callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    async def close(self) -> None:
        """Close Realtime connection."""
        self._is_connected = False
        if self._connection_cm:
            try:
                await self._connection_cm.__aexit__(None, None, None)
                logger.info("OpenAI Realtime disconnected")
            except Exception as e:
                logger.error(f"Error closing OpenAI Realtime: {e}")
        self._connection_cm = None
        self.conn = None

    def _should_translate(self) -> bool:
        target = self.target_language
        if not target or target == "auto":
            return False
        source = self.source_language or self.last_transcript_language
        if source and source == target:
            return False
        return True

    async def _translate_if_needed(self, text: str) -> str | None:
        if not self._should_translate():
            return None
        target = self.target_language
        try:
            response = await self.client.responses.create(
                model=settings.translation_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Translate the user's transcription to the target language. "
                            "Return only the translated text without extra commentary."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Target language: {target}\nText: {text}",
                    },
                ],
            )
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            await self._safe_callback(self.on_error, f"Translation failed: {e}")
            return None

        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()

        output = getattr(response, "output", None)
        if output:
            try:
                parts = []
                for item in output:
                    for content in getattr(item, "content", []) or []:
                        text_value = getattr(content, "text", None)
                        if text_value:
                            parts.append(text_value)
                if parts:
                    return "".join(parts).strip()
            except Exception:
                return None

        return None

    @property
    def is_connected(self) -> bool:
        """Check if connected to OpenAI."""
        return self._is_connected

    @staticmethod
    def _get_value(obj: Any, name: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)
