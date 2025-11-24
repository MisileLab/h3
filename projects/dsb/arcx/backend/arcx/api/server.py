"""FastAPI server for overlay communication"""

import base64
import logging
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from arcx.api.schemas import (
    ConfigUpdate,
    EVResponse,
    FeedbackSubmission,
    StartRunRequest,
    EndRunRequest,
    SystemStatus,
    ValuateScreenshotRequest,
    AutoValuationResult,
    DetectedItemSchema,
)
from arcx.config import config
from arcx.valuation import ItemValuator

logger = logging.getLogger(__name__)


# Global state (will be initialized properly in Phase 3)
class AppState:
    """Application state container"""

    def __init__(self):
        self.model = None  # EVModel instance
        self.inference_engine = None  # InferenceEngine instance
        self.data_logger = None  # DataLogger instance
        self.valuator = None  # ItemValuator instance
        self.current_run_id = None
        self.is_capturing = False


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    logger.info("Starting ArcX API server...")

    # Initialize model and inference engine
    from arcx.ml.model import EVModel
    from arcx.ml.inference import InferenceEngine
    from arcx.device import device_manager

    logger.info("Loading EVModel...")
    try:
        app_state.model = EVModel(
            encoder_backbone=config.model.encoder_backbone,
            latent_dim=config.model.encoder_dim,
            encoder_pretrained=True,
            hidden_dim=config.model.hidden_dim,
            num_quantiles=config.model.num_quantiles,
            temporal_encoder=config.model.temporal_encoder,
        )
        app_state.model.to(device_manager.device)
        app_state.model.eval()

        # Try to load weights if available
        if config.model.encoder_path.exists() and config.model.qnet_path.exists():
            from arcx.ml.utils import load_model_safetensors

            logger.info("Loading trained weights...")
            # Load encoder
            load_model_safetensors(
                app_state.model.encoder, config.model.encoder_path, device=device_manager.device
            )
            # Load qnet
            load_model_safetensors(
                app_state.model.qnet, config.model.qnet_path, device=device_manager.device
            )
            logger.info("Trained weights loaded successfully")
        else:
            logger.warning("No trained weights found, using untrained model")

        logger.info(f"Model loaded on device: {device_manager.device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise

    logger.info("Creating InferenceEngine...")
    try:
        app_state.inference_engine = InferenceEngine(
            model=app_state.model,
            capture_fps=config.capture.fps,
            inference_interval=config.inference.update_interval,
            sequence_length=config.model.sequence_length,
        )
        logger.info("InferenceEngine created successfully")
    except Exception as e:
        logger.error(f"Failed to create inference engine: {e}", exc_info=True)
        raise

    logger.info("Creating ItemValuator...")
    try:
        app_state.valuator = ItemValuator(
            model_path=None,  # Will use default path or placeholder
            confidence=0.5,
            device=str(device_manager.device),
            game_phase="mid_wipe",
        )
        logger.info("ItemValuator created successfully")
    except Exception as e:
        logger.warning(f"Failed to create valuator: {e}", exc_info=True)
        logger.warning("Item valuation will be unavailable")

    logger.info("API server ready")
    yield

    logger.info("Shutting down ArcX API server...")
    # Cleanup
    if app_state.inference_engine:
        app_state.inference_engine.stop()
        logger.info("InferenceEngine stopped")


app = FastAPI(
    title="ArcX EV Overlay API",
    version="0.1.0",
    description="Real-time EV calculation for extraction shooters",
    lifespan=lifespan,
)

# CORS middleware for overlay clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check"""
    return {"status": "ok", "service": "arcx-api", "version": "0.1.0"}


@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    from arcx.device import device_manager

    stats = {}
    if app_state.inference_engine:
        stats = app_state.inference_engine.get_stats()

    return SystemStatus(
        is_capturing=stats.get("is_running", False),
        is_model_loaded=app_state.model is not None,
        buffer_frames=stats.get("buffer_size", 0),
        device_backend=device_manager.backend.value,
        fps=stats.get("capture_fps", 0.0),
        inference_time_ms=stats.get("avg_inference_time_ms", 0.0),
    )


@app.get("/ev", response_model=EVResponse)
async def get_ev():
    """
    Get current Expected Value recommendation.

    This is the main endpoint called by the overlay at regular intervals.
    """
    if not app_state.model or not app_state.inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    stats = app_state.inference_engine.get_stats()
    if not stats.get("is_running"):
        raise HTTPException(status_code=400, detail="Screen capture not active")

    # Get latest prediction
    prediction = app_state.inference_engine.get_latest_prediction()
    if prediction is None:
        raise HTTPException(
            status_code=503, detail="No prediction available yet (buffer filling)"
        )

    # Determine message and color based on recommendation and delta
    delta = prediction.delta_ev
    rec = prediction.recommendation

    if rec == "stay":
        if delta > 50:
            message = "지금은 파밍 계속 강력 권장 (EV 매우 높음)"
        else:
            message = "지금은 파밍 계속 권장 (EV 높음)"
        color = "green"
    elif rec == "extract":
        if delta < -50:
            message = "지금은 추출 강력 권장 (위험도 높음)"
        else:
            message = "지금은 추출 권장 (위험 대비 낮은 EV)"
        color = "red"
    else:
        message = "둘 다 비슷함, 판단에 맡김"
        color = "yellow"

    return EVResponse(
        ev_stay=prediction.ev_stay,
        ev_extract=prediction.ev_extract,
        delta_ev=prediction.delta_ev,
        recommendation=prediction.recommendation,
        confidence=prediction.confidence,
        message=message,
        color=color,
        timestamp=prediction.timestamp,
        risk_profile=config.inference.risk_profile,
    )


@app.post("/config")
async def update_config(update: ConfigUpdate):
    """Update runtime configuration"""
    config.inference.risk_profile = update.risk_profile
    logger.info(f"Risk profile updated to: {update.risk_profile}")
    return {"status": "ok", "risk_profile": update.risk_profile}


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    """Submit user feedback on a recommendation"""
    logger.info(f"Feedback received: {feedback.rating} for run {feedback.run_id}")

    # TODO: Log to Parquet in Phase 4
    # app_state.data_logger.log_feedback(feedback)

    return {"status": "ok"}


@app.post("/run/start")
async def start_run(request: StartRunRequest):
    """Start a new run/raid"""
    import uuid

    if not app_state.inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    run_id = str(uuid.uuid4())
    app_state.current_run_id = run_id

    # Start inference engine if not already running
    if not app_state.inference_engine.get_stats().get("is_running"):
        app_state.inference_engine.start()
        logger.info("InferenceEngine started")

    logger.info(f"Started run: {run_id}, map: {request.map_id}")

    # TODO: Initialize data logger for this run in Phase 4
    # app_state.data_logger.start_run(run_id, request.map_id)

    return {"status": "ok", "run_id": run_id}


@app.post("/valuate", response_model=AutoValuationResult)
async def valuate_screenshot(request: ValuateScreenshotRequest):
    """Valuate items from extraction screenshot using YOLO"""
    if not app_state.valuator:
        raise HTTPException(
            status_code=503,
            detail="Item valuator not initialized"
        )

    try:
        # Decode base64 screenshot
        img_data = base64.b64decode(request.screenshot_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        screenshot = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if screenshot is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image data"
            )

        # Update game phase if provided
        if request.game_phase:
            app_state.valuator.set_game_phase(request.game_phase)

        # Perform valuation
        result = app_state.valuator.valuate_screenshot(screenshot)

        # Convert to API schema
        items = [
            DetectedItemSchema(
                item_type=item.item_type,
                rarity=item.rarity,
                confidence=item.confidence,
                estimated_value=item.estimated_value,
                bbox=item.bbox,
            )
            for item in result.items
        ]

        return AutoValuationResult(
            total_value=result.total_value,
            items=items,
            num_items=result.num_items,
            avg_confidence=result.avg_confidence,
            value_breakdown=result.value_breakdown,
            rarity_counts=result.rarity_counts,
            phase_multiplier=result.phase_multiplier,
        )

    except Exception as e:
        logger.error(f"Valuation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Valuation failed: {str(e)}"
        )


@app.post("/run/end")
async def end_run(request: EndRunRequest):
    """End current run with results"""
    if app_state.current_run_id != request.run_id:
        raise HTTPException(status_code=400, detail="Run ID mismatch")

    total_value = request.auto_valuation.total_value
    num_items = request.auto_valuation.num_items

    logger.info(
        f"Ended run: {request.run_id}, "
        f"loot: {total_value:.2f} ({num_items} items detected), "
        f"time: {request.total_time_sec}s, "
        f"success: {request.success}"
    )

    # TODO: Finalize and save run data in Phase 4
    # app_state.data_logger.end_run(
    #     final_loot_value=total_value,
    #     total_time_sec=request.total_time_sec,
    #     success=request.success
    # )

    app_state.current_run_id = None

    # Optionally stop inference engine
    # (keep running for now to allow immediate next run)

    return {"status": "ok"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level="info",
    )
