import asyncio
import hashlib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

from app.audio_cache import AudioCache
from app.config import settings
from app.auth import validate_token, hash_token
from app.meter import UsageMeter
from app.openai_realtime import OpenAIRealtimeClient
from app.protocol import (
    StartMessage,
    PartialTranscriptMessage,
    FinalTranscriptMessage,
    InfoMessage,
    ErrorMessage,
)
from app.logging_setup import logger

cache = AudioCache("data/audio_cache.sqlite3")

app = FastAPI(title="StreamIt Real-time Caption Proxy")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"ok": True}


@app.websocket("/ws/viewer")
async def websocket_viewer(
    websocket: WebSocket,
    token: str = Query(..., description="Bearer token for authentication"),
):
    """WebSocket endpoint for caption streaming."""

    await websocket.accept()
    logger.info(f"WebSocket connection from token: {hash_token(token)}")

    if not validate_token(token):
        logger.warning(f"Rejecting invalid token: {hash_token(token)}")
        await websocket.close(code=1008, reason="Invalid token")
        return

    if not UsageMeter.check_limit(token, settings.max_seconds_per_user):
        await websocket.send_json(
            ErrorMessage(
                message="Daily usage limit exceeded", code="LIMIT_EXCEEDED"
            ).model_dump()
        )
        await websocket.close(code=1008, reason="Limit exceeded")
        return

    openai_client: OpenAIRealtimeClient | None = None
    total_bytes = 0
    language = "auto"
    target_language = "auto"
    session_active = False

    async def on_transcript_delta(text: str):
        """Forward partial transcript to viewer."""
        try:
            await websocket.send_json(PartialTranscriptMessage(text=text).model_dump())
        except Exception as e:
            logger.error(f"Failed to send partial: {e}")

    async def on_transcript_completed(text: str):
        """Forward final transcript to viewer."""
        try:
            await websocket.send_json(FinalTranscriptMessage(text=text).model_dump())
            if openai_client and openai_client.current_audio_hash:
                cache.set(
                    openai_client.current_audio_hash,
                    target_language,
                    text,
                )
                openai_client.current_audio_hash = None
        except Exception as e:
            logger.error(f"Failed to send final: {e}")

    async def on_openai_error(message: str):
        """Forward OpenAI error to viewer."""
        try:
            await websocket.send_json(ErrorMessage(message=message).model_dump())
        except Exception as e:
            logger.error(f"Failed to send error: {e}")

    async def on_translation(text: str):
        """Forward translated transcript to viewer."""
        try:
            await websocket.send_json(FinalTranscriptMessage(text=text).model_dump())
            if openai_client and openai_client.current_audio_hash:
                cache.set(
                    openai_client.current_audio_hash,
                    target_language,
                    text,
                )
                openai_client.current_audio_hash = None
        except Exception as e:
            logger.error(f"Failed to send translation: {e}")

    try:
        openai_client = OpenAIRealtimeClient(
            on_transcript_delta=on_transcript_delta,
            on_transcript_completed=on_transcript_completed,
            on_translation=on_translation,
            on_error=on_openai_error,
        )

        async def handle_viewer_messages():
            """Handle messages from viewer extension."""
            nonlocal language, target_language, session_active, total_bytes

            try:
                async for message in websocket.iter_json():
                    msg_type = message.get("type")

                    if msg_type == "start":
                        try:
                            start_msg = StartMessage(**message)
                            language = start_msg.lang
                            target_language = start_msg.targetLang
                            logger.info(
                                "Start request: %s, lang=%s, target=%s, platform=%s",
                                start_msg.clientSessionId,
                                language,
                                target_language,
                                start_msg.platformHint,
                            )

                            if not session_active:
                                success = await openai_client.connect(
                                    language, target_language
                                )
                                if success:
                                    session_active = True

                                    asyncio.create_task(openai_client.listen())
                                    await websocket.send_json(
                                        InfoMessage(
                                            message="Connected to transcription service"
                                        ).model_dump()
                                    )
                                else:
                                    await websocket.send_json(
                                        ErrorMessage(
                                            message="Failed to connect to OpenAI"
                                        ).model_dump()
                                    )

                        except Exception as e:
                            logger.error(f"Start message error: {e}")
                            await websocket.send_json(
                                ErrorMessage(message=f"Invalid start: {e}").model_dump()
                            )

                    elif msg_type == "stop":
                        logger.info("Stop request received")
                        session_active = False
                        await openai_client.close()

                        seconds_used = total_bytes / 48000.0
                        UsageMeter.record_session(token, seconds_used)
                        await websocket.send_json(
                            InfoMessage(
                                message="Session stopped", secondsUsed=seconds_used
                            ).model_dump()
                        )

                    else:
                        logger.warning(f"Unknown message type: {msg_type}")

            except WebSocketDisconnect:
                logger.info("Viewer disconnected")
            except Exception as e:
                logger.error(f"Viewer message handler error: {e}")

        async def handle_audio_frames():
            """Handle binary audio frames from viewer."""
            nonlocal total_bytes, session_active

            try:
                async for message in websocket.iter_bytes():
                    if session_active and openai_client.is_connected:
                        audio_hash = hashlib.blake2b(
                            message, digest_size=16
                        ).hexdigest()
                        cached = cache.get(audio_hash, target_language)
                        if cached:
                            await websocket.send_json(
                                FinalTranscriptMessage(text=cached).model_dump()
                            )
                        else:
                            openai_client.current_audio_hash = audio_hash
                            await openai_client.send_audio(message)
                        total_bytes += len(message)

                        UsageMeter.add_bytes(token, len(message))

                        if not UsageMeter.check_limit(
                            token, settings.max_seconds_per_user
                        ):
                            await websocket.send_json(
                                ErrorMessage(
                                    message="Daily usage limit exceeded",
                                    code="LIMIT_EXCEEDED",
                                ).model_dump()
                            )
                            session_active = False
                            await openai_client.close()

            except WebSocketDisconnect:
                logger.info("Viewer audio stream ended")
            except Exception as e:
                logger.error(f"Audio frame handler error: {e}")

        await asyncio.gather(
            handle_viewer_messages(),
            handle_audio_frames(),
        )

    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
        await websocket.send_json(
            ErrorMessage(message=f"Server error: {e}").model_dump()
        )

    finally:
        if openai_client:
            await openai_client.close()

        if total_bytes > 0:
            seconds_used = total_bytes / 48000.0
            UsageMeter.record_session(token, seconds_used)

        logger.info(f"Connection closed for token: {hash_token(token)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
