"""
Enhanced FastAPI application with comprehensive features.
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
import logging
import uvicorn
from datetime import datetime
import json
import os
from pathlib import Path

from src.predict import ModelPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Pipeline API",
    description="A comprehensive MLOps pipeline with model training and prediction capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    feature1: float = Field(..., description="First feature value")
    feature2: float = Field(..., description="Second feature value")
    feature3: float = Field(..., description="Third feature value")
    
    @validator('feature1', 'feature2', 'feature3')
    def validate_features(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('Feature must be a number')
        return float(v)

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    data: List[Dict[str, float]] = Field(..., description="List of feature dictionaries")

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: float = Field(..., description="Probability of positive class")
    model_type: str = Field(..., description="Type of model used")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[int] = Field(..., description="List of predicted classes")
    probabilities: List[List[float]] = Field(..., description="List of prediction probabilities")
    model_type: str = Field(..., description="Type of model used")
    n_samples: int = Field(..., description="Number of samples processed")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_info: Optional[Dict] = Field(None, description="Model information")

# Dependency to get predictor instance
def get_predictor() -> ModelPredictor:
    """Get the global predictor instance."""
    global predictor
    if predictor is None:
        try:
            predictor = ModelPredictor()
            logger.info("Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise HTTPException(status_code=500, detail="Model not available")
    return predictor

# API Routes
@app.get("/", response_model=Dict[str, str])
async def home():
    """Home endpoint with basic information."""
    return {
        "message": "MLOps Pipeline API is running 🚀",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        pred = get_predictor()
        model_info = pred.get_model_info()
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=True,
            model_info=model_info
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=False,
            model_info={"error": str(e)}
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest, pred: ModelPredictor = Depends(get_predictor)):
    """Make a single prediction."""
    try:
        # Convert request to dict
        data = request.dict()
        
        # Make prediction
        result = pred.predict(data)
        
        return PredictionResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            model_type=result['model_type'],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, pred: ModelPredictor = Depends(get_predictor)):
    """Make batch predictions."""
    try:
        # Make predictions
        result = pred.predict(request.data)
        
        return BatchPredictionResponse(
            predictions=result['predictions'],
            probabilities=result['probabilities'],
            model_type=result['model_type'],
            n_samples=result['n_samples'],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/info")
async def get_model_info(pred: ModelPredictor = Depends(get_predictor)):
    """Get information about the loaded model."""
    try:
        return pred.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/metrics")
async def get_model_metrics():
    """Get model training metrics if available."""
    try:
        metadata_path = "models/metadata.json"
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return {
                "training_metrics": metadata,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {"message": "No training metrics available"}
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "ValueError"}
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "type": "FileNotFoundError"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting MLOps Pipeline API...")
    try:
        # Initialize predictor
        get_predictor()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"API startup failed: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down MLOps Pipeline API...")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
