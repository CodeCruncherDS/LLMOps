"""
FastAPI Web Service Module

Provides REST API endpoints for model inference, health checks,
and model versioning with rate limiting and CORS support.
"""

import logging
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import settings
from src.inference import InferenceError, LLMInference
from src.model import ModelVersionManager
from src.monitor import HealthChecker, PredictionMonitor

logger = logging.getLogger(__name__)


# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# Global instances (initialized in lifespan)
inference_engine: Optional[LLMInference] = None
monitor: Optional[PredictionMonitor] = None
health_checker: Optional[HealthChecker] = None


# Request/Response Models
class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Input text for prediction",
        json_schema_extra={"example": "This is a sample text for classification."}
    )
    return_all_scores: bool = Field(
        default=False,
        description="Return scores for all classes (classification only)"
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=4096,
        description="Maximum new tokens to generate (generation only)"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (generation only)"
    )


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    
    input_text: str
    output: Any
    model_version: str
    inference_time_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class BatchPredictRequest(BaseModel):
    """Request model for batch prediction endpoint."""
    
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts for batch prediction"
    )
    return_all_scores: bool = Field(default=False)


class BatchPredictResponse(BaseModel):
    """Response model for batch prediction endpoint."""
    
    predictions: list[PredictResponse]
    total_count: int
    successful_count: int
    batch_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str
    timestamp: str
    version: str
    environment: str
    checks: Optional[dict[str, Any]] = None


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint."""
    
    model_version: str
    model_path: str
    task_type: str
    device: str
    model_type: str
    config: dict[str, Any]


class ModelVersionsResponse(BaseModel):
    """Response model for model versions endpoint."""
    
    versions: list[str]
    latest_version: Optional[str]
    metadata: dict[str, Any]


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    global inference_engine, monitor, health_checker
    
    # Startup
    logger.info("Starting LLM Pipeline API...")
    
    try:
        # Initialize components
        monitor = PredictionMonitor()
        health_checker = HealthChecker()
        
        # Initialize inference engine (lazy loading)
        inference_engine = LLMInference(
            task_type="classification",
            enable_monitoring=True,
        )
        
        # Ensure directories exist
        settings.ensure_directories()
        
        logger.info(
            f"API started successfully. Environment: {settings.environment.value}"
        )
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down LLM Pipeline API...")
        
        if inference_engine:
            inference_engine.unload_model()
        
        logger.info("API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.project_name,
    version=settings.project_version,
    description="Production-ready MLOps API for Large Language Models",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=settings.api.cors_allow_credentials,
    allow_methods=settings.api.cors_allow_methods,
    allow_headers=settings.api.cors_allow_headers,
)


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    response = await call_next(request)
    
    latency_ms = (time.time() - start_time) * 1000
    
    if monitor:
        monitor.log_request(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=response.status_code,
            latency_ms=latency_ms,
            client_ip=request.client.host if request.client else None,
        )
    
    return response


# Exception handlers
@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError):
    """Handle inference errors."""
    logger.error(f"Inference error: {exc.message}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Inference Error",
            detail=exc.message,
        ).model_dump(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
        ).model_dump(),
    )


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.project_name,
        "version": settings.project_version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
)
async def health_check():
    """
    Check API and system health.
    
    Returns health status of the API and its components.
    """
    checks = None
    overall_status = "healthy"
    
    if health_checker:
        full_check = health_checker.full_health_check(inference_engine)
        checks = full_check
        overall_status = full_check.get("overall_status", "healthy")
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        version=settings.project_version,
        environment=settings.environment.value,
        checks=checks,
    )


@app.get(
    "/health/live",
    tags=["Health"],
    summary="Liveness probe",
)
async def liveness_probe():
    """Kubernetes-style liveness probe."""
    return {"status": "alive"}


@app.get(
    "/health/ready",
    tags=["Health"],
    summary="Readiness probe",
)
async def readiness_probe():
    """Kubernetes-style readiness probe."""
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )
    
    try:
        # Try to get model info as a readiness check
        info = inference_engine.get_model_info()
        return {"status": "ready", "model_version": info.get("model_version")}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not ready: {str(e)}",
        )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Make a single prediction",
)
@limiter.limit(f"{settings.api.rate_limit_requests}/minute")
async def predict(request: Request, body: PredictRequest):
    """
    Make a prediction for a single text input.
    
    Supports both classification and generation tasks based on the loaded model.
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not available",
        )
    
    # Build kwargs for generation
    gen_kwargs = {}
    if body.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = body.max_new_tokens
    if body.temperature is not None:
        gen_kwargs["temperature"] = body.temperature
    
    result = inference_engine.predict(
        text=body.text,
        return_all_scores=body.return_all_scores,
        **gen_kwargs,
    )
    
    return PredictResponse(
        input_text=result.input_text,
        output=result.output,
        model_version=result.model_version,
        inference_time_ms=result.inference_time_ms,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Inference"],
    summary="Make batch predictions",
)
@limiter.limit(f"{settings.api.rate_limit_requests // 10}/minute")
async def predict_batch(request: Request, body: BatchPredictRequest):
    """
    Make predictions for multiple text inputs.
    
    Limited to 100 texts per request.
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not available",
        )
    
    start_time = time.time()
    
    results = inference_engine.predict_batch(
        texts=body.texts,
        return_all_scores=body.return_all_scores,
    )
    
    batch_time_ms = (time.time() - start_time) * 1000
    
    predictions = [
        PredictResponse(
            input_text=r.input_text,
            output=r.output,
            model_version=r.model_version,
            inference_time_ms=r.inference_time_ms,
        )
        for r in results
    ]
    
    successful = sum(
        1 for r in results
        if not (isinstance(r.output, dict) and "error" in r.output)
    )
    
    return BatchPredictResponse(
        predictions=predictions,
        total_count=len(body.texts),
        successful_count=successful,
        batch_time_ms=batch_time_ms,
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get model information",
)
async def get_model_info():
    """
    Get information about the currently loaded model.
    
    Returns model version, type, device, and configuration.
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not available",
        )
    
    info = inference_engine.get_model_info()
    
    return ModelInfoResponse(
        model_version=info["model_version"],
        model_path=info["model_path"],
        task_type=info["task_type"],
        device=info["device"],
        model_type=info["model_type"],
        config={
            "vocab_size": info.get("vocab_size"),
            "max_position_embeddings": info.get("max_position_embeddings"),
            "hidden_size": info.get("hidden_size"),
            "num_labels": info.get("num_labels"),
        },
    )


@app.get(
    "/model/versions",
    response_model=ModelVersionsResponse,
    tags=["Model"],
    summary="List model versions",
)
async def list_model_versions():
    """
    List all available model versions.
    
    Returns version list, latest version, and metadata.
    """
    version_manager = ModelVersionManager()
    
    versions = version_manager.list_versions()
    latest = version_manager.get_latest_version()
    metadata = version_manager.get_metadata()
    
    return ModelVersionsResponse(
        versions=versions,
        latest_version=latest,
        metadata=metadata,
    )


@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Get monitoring metrics",
)
async def get_metrics():
    """
    Get current monitoring metrics.
    
    Returns prediction counts, latency stats, and error rates.
    """
    if monitor is None:
        return {"message": "Monitoring not enabled"}
    
    return monitor.get_metrics()


@app.post(
    "/metrics/reset",
    tags=["Monitoring"],
    summary="Reset monitoring metrics",
)
async def reset_metrics():
    """Reset all monitoring metrics to zero."""
    if monitor is None:
        return {"message": "Monitoring not enabled"}
    
    monitor.reset_metrics()
    return {"message": "Metrics reset successfully"}


# Development-only endpoints
if settings.is_development:
    @app.post(
        "/model/reload",
        tags=["Development"],
        summary="Reload model (dev only)",
    )
    async def reload_model():
        """Reload the model from disk (development only)."""
        global inference_engine
        
        if inference_engine:
            inference_engine.unload_model()
            inference_engine._load_model()
        
        return {"message": "Model reloaded successfully"}


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.is_development,
        workers=settings.api.workers,
    )
