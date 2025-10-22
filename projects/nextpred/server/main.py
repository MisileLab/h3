"""
FastAPI Server for Next Action Predictor
Handles event collection, model serving, and user feedback
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import hashlib
import json

from .db.postgres import get_db, Database
from .db.qdrant_client import QdrantManager
from .services.data_collector import DataCollector
from .services.predictor import Predictor
from .services.model_exporter import ModelExporter
from .monitoring.metrics import MetricsCollector
from .models.schemas import (
    EventBatch, EventResponse, ModelVersionResponse, 
    FeedbackRequest, FeedbackResponse, HealthResponse,
    UserStatsResponse, PredictionRequest, PredictionResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
db: Database = None
qdrant: QdrantManager = None
data_collector: DataCollector = None
predictor: Predictor = None
model_exporter: ModelExporter = None
metrics: MetricsCollector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Next Action Predictor server...")
    
    global db, qdrant, data_collector, predictor, model_exporter, metrics
    
    try:
        # Initialize database connections
        db = Database()
        await db.initialize()
        
        # Initialize Qdrant vector database
        qdrant = QdrantManager()
        await qdrant.initialize()
        
        # Initialize services
        data_collector = DataCollector(db, qdrant)
        predictor = Predictor(db, qdrant)
        model_exporter = ModelExporter(db)
        metrics = MetricsCollector()
        
        # Start background tasks
        asyncio.create_task(metrics.start_collection())
        
        logger.info("Server initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")
    if db:
        await db.close()
    if qdrant:
        await qdrant.close()
    logger.info("Server shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Next Action Predictor API",
    description="ML-powered next action prediction for Chrome tabs",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*", "moz-extension://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database connection
async def get_database() -> Database:
    return db

# Health check endpoint
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check server health and component status"""
    try:
        db_status = await db.health_check() if db else False
        qdrant_status = await qdrant.health_check() if qdrant else False
        
        return HealthResponse(
            healthy=True,
            server="nextpred-api",
            version="1.0.0",
            timestamp=datetime.utcnow(),
            components={
                "database": "healthy" if db_status else "unhealthy",
                "qdrant": "healthy" if qdrant_status else "unhealthy",
                "predictor": "healthy" if predictor else "unhealthy"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            healthy=False,
            server="nextpred-api",
            version="1.0.0",
            timestamp=datetime.utcnow(),
            error=str(e)
        )

# Event endpoints
@app.post("/api/events/batch", response_model=EventResponse)
async def upload_events(
    events: EventBatch,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_database)
):
    """Upload a batch of events from the Chrome extension"""
    try:
        # Validate events
        if not events.events or len(events.events) == 0:
            raise HTTPException(status_code=400, detail="No events provided")
        
        if len(events.events) > 1000:
            raise HTTPException(status_code=400, detail="Too many events (max 1000)")
        
        # Store events in database
        stored_count = await data_collector.store_events(events.events)
        
        # Process events in background (embeddings, analysis, etc.)
        background_tasks.add_task(
            data_collector.process_events_batch, 
            events.events
        )
        
        # Update metrics
        metrics.record_events_uploaded(len(events.events))
        
        logger.info(f"Stored {stored_count} events from user")
        
        return EventResponse(
            success=True,
            stored=stored_count,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to store events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/events/stats", response_model=UserStatsResponse)
async def get_user_stats(
    user_id: Optional[str] = None,
    days: int = 7,
    db: Database = Depends(get_database)
):
    """Get user statistics and activity summary"""
    try:
        stats = await data_collector.get_user_stats(user_id, days)
        return UserStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get user stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model endpoints
@app.get("/api/model/version", response_model=ModelVersionResponse)
async def get_model_version():
    """Get current model version information"""
    try:
        version_info = await model_exporter.get_latest_version()
        return ModelVersionResponse(**version_info)
    except Exception as e:
        logger.error(f"Failed to get model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/download")
async def download_model(version: Optional[str] = None):
    """Download ONNX model file"""
    try:
        if not version:
            version_info = await model_exporter.get_latest_version()
            version = version_info["version"]
        
        model_path = await model_exporter.get_model_file(version)
        if not model_path:
            raise HTTPException(status_code=404, detail="Model version not found")
        
        # Generate download URL or serve file directly
        return FileResponse(
            path=model_path,
            filename=f"nextpred_model_v{version}.onnx",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/upload")
async def upload_model(
    file: UploadFile = File(...),
    version: str = None,
    db: Database = Depends(get_database)
):
    """Upload a new model version (admin endpoint)"""
    try:
        if not version:
            raise HTTPException(status_code=400, detail="Version parameter required")
        
        # Validate file type
        if not file.filename.endswith('.onnx'):
            raise HTTPException(status_code=400, detail="Only ONNX files are supported")
        
        # Store model file
        model_info = await model_exporter.store_model(file, version)
        
        logger.info(f"Model version {version} uploaded successfully")
        
        return {"success": True, "model_info": model_info}
        
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prediction endpoints
@app.post("/api/predict", response_model=PredictionResponse)
async def predict_next_action(
    request: PredictionRequest,
    db: Database = Depends(get_database)
):
    """Get next action predictions for current context"""
    try:
        predictions = await predictor.predict(
            context=request.context,
            current_tab=request.current_tab
        )
        
        # Update metrics
        metrics.record_prediction_request()
        
        return PredictionResponse(
            predictions=predictions,
            timestamp=datetime.utcnow(),
            model_version=await predictor.get_model_version()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Feedback endpoints
@app.post("/api/feedback", response_model=FeedbackResponse)
async def record_feedback(
    feedback: FeedbackRequest,
    db: Database = Depends(get_database)
):
    """Record user feedback on predictions"""
    try:
        feedback_id = await data_collector.store_feedback(feedback)
        
        # Update metrics
        metrics.record_feedback(feedback.feedback_type)
        
        logger.info(f"Feedback recorded: {feedback_id}")
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feedback/stats")
async def get_feedback_stats(
    days: int = 7,
    db: Database = Depends(get_database)
):
    """Get feedback statistics and accuracy metrics"""
    try:
        stats = await data_collector.get_feedback_stats(days)
        return stats
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Metrics endpoints
@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    try:
        metrics_data = await metrics.get_prometheus_metrics()
        return JSONResponse(
            content=metrics_data,
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Admin endpoints
@app.get("/api/admin/status")
async def admin_status():
    """Admin endpoint for system status"""
    try:
        status = {
            "server": {
                "uptime": "active",
                "version": "1.0.0",
                "timestamp": datetime.utcnow().isoformat()
            },
            "database": await db.get_status() if db else "unavailable",
            "qdrant": await qdrant.get_status() if qdrant else "unavailable",
            "model": await model_exporter.get_status() if model_exporter else "unavailable",
            "metrics": await metrics.get_summary() if metrics else "unavailable"
        }
        
        return status
    except Exception as e:
        logger.error(f"Admin status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/train")
async def trigger_training(
    background_tasks: BackgroundTasks,
    force: bool = False
):
    """Trigger model training (admin endpoint)"""
    try:
        from .training.train_daily import run_training
        
        # Run training in background
        background_tasks.add_task(run_training, force=force)
        
        return {"success": True, "message": "Training started"}
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )