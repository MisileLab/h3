"""
FastAPI server for ELECTRA jailbreak detection.
"""

import time
import os
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn

from ..models.electra_classifier import ElectraJailbreakClassifier
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    SafetyDecision
)


class JailbreakGuardAPI:
    """FastAPI server for jailbreak detection."""
    
    def __init__(self, model_path: str, use_lora: bool = False):
        self.app = FastAPI(
            title="ELECTRA Jailbreak Guard API",
            description="API for detecting prompt injection and jailbreak attempts",
            version="0.1.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Load model
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.use_lora = use_lora
        self.load_model()
        
        # Setup routes
        self.setup_routes()
    
    def load_model(self):
        """Load the trained model."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model, self.tokenizer = ElectraJailbreakClassifier.load_model(
                self.model_path, 
                use_lora=self.use_lora
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                model_loaded=self.model is not None,
                version="0.1.0"
            )
        
        @self.app.get("/model/info", response_model=ModelInfoResponse)
        async def model_info():
            """Get model information."""
            if not self.model:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not loaded"
                )
            
            trainable_params, total_params = self.model.get_trainable_parameters()
            
            return ModelInfoResponse(
                model_name=self.model.model_name,
                model_type="lora" if self.use_lora else "full",
                num_parameters=total_params,
                max_length=256,
                version="0.1.0"
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Predict jailbreak for a single text."""
            if not self.model:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not loaded"
                )
            
            start_time = time.time()
            
            try:
                # Make prediction
                result = self.model.predict_single(
                    request.text,
                    self.tokenizer,
                    max_length=256
                )
                
                # Determine safety decision
                unsafe_prob = result["unsafe_probability"]
                if unsafe_prob > 0.7:
                    decision = SafetyDecision.BLOCK
                elif unsafe_prob > 0.4:
                    decision = SafetyDecision.REVIEW
                else:
                    decision = SafetyDecision.SAFE
                
                processing_time = time.time() - start_time
                
                response = PredictionResponse(
                    text=request.text,
                    predicted_class=result["predicted_class"],
                    decision=decision,
                    confidence=result["confidence"],
                    unsafe_probability=result["unsafe_probability"],
                    safe_probability=result["safe_probability"],
                    probabilities={
                        "safe": result["safe_probability"],
                        "unsafe": result["unsafe_probability"]
                    } if request.return_probabilities else None,
                    processing_time=processing_time
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Prediction failed: {str(e)}"
                )
        
        @self.app.post("/predict/batch", response_model=BatchPredictionResponse)
        async def predict_batch(request: BatchPredictionRequest):
            """Predict jailbreak for multiple texts."""
            if not self.model:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not loaded"
                )
            
            if len(request.texts) > 100:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Maximum 100 texts allowed per batch"
                )
            
            start_time = time.time()
            results = []
            
            try:
                for text in request.texts:
                    # Make prediction
                    result = self.model.predict_single(
                        text,
                        self.tokenizer,
                        max_length=256
                    )
                    
                    # Determine safety decision
                    unsafe_prob = result["unsafe_probability"]
                    if unsafe_prob > 0.7:
                        decision = SafetyDecision.BLOCK
                    elif unsafe_prob > 0.4:
                        decision = SafetyDecision.REVIEW
                    else:
                        decision = SafetyDecision.SAFE
                    
                    response = PredictionResponse(
                        text=text,
                        predicted_class=result["predicted_class"],
                        decision=decision,
                        confidence=result["confidence"],
                        unsafe_probability=result["unsafe_probability"],
                        safe_probability=result["safe_probability"],
                        probabilities={
                            "safe": result["safe_probability"],
                            "unsafe": result["unsafe_probability"]
                        } if request.return_probabilities else None,
                        processing_time=0  # Will be updated with total time
                    )
                    
                    results.append(response)
                
                total_time = time.time() - start_time
                
                # Update processing times
                avg_time = total_time / len(results)
                for result in results:
                    result.processing_time = avg_time
                
                return BatchPredictionResponse(
                    results=results,
                    total_processed=len(results),
                    processing_time=total_time
                )
                
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Batch prediction failed: {str(e)}"
                )
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server."""
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def create_app(model_path: str, use_lora: bool = False) -> FastAPI:
    """Create FastAPI app instance."""
    api = JailbreakGuardAPI(model_path, use_lora)
    return api.app


if __name__ == "__main__":
    # Example usage
    model_path = os.environ.get("MODEL_PATH", "./checkpoints/final_model")
    use_lora = os.environ.get("USE_LORA", "false").lower() == "true"
    
    api = JailbreakGuardAPI(model_path, use_lora)
    api.run()