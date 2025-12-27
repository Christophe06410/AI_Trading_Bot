#!/usr/bin/env python3
"""
AI Recommendation Service for Trading Bot
FastAPI service that provides trading signals
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import asyncio

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
import uvicorn

# Local imports
from src.ml_model import MLModel, PredictionResult
from src.data_store import DataStore
from src.recommendation_logic import RecommendationEngine
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ai_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================

class Candle(BaseModel):
    """OHLCV candle with technical indicators"""
    timestamp: str = Field(..., description="ISO format timestamp")
    o: float = Field(..., description="Open price", gt=0)
    h: float = Field(..., description="High price", gt=0)
    l: float = Field(..., description="Low price", gt=0)
    c: float = Field(..., description="Close price", gt=0)
    v: float = Field(..., description="Volume", ge=0)
    ema200: Optional[float] = Field(None, description="200-period EMA")
    
    @validator('h')
    def validate_high(cls, v, values):
        if 'l' in values and v < values['l']:
            raise ValueError('High must be >= low')
        return v
    
    @validator('c')
    def validate_close(cls, v, values):
        if 'o' in values and v <= 0:
            raise ValueError('Close must be > 0')
        return v


class Position(BaseModel):
    """Trading position"""
    id: Optional[str] = Field(None, description="Position ID")
    timestamp: str = Field(..., description="Entry time ISO format")
    direction: str = Field(..., description="LONG or SHORT")
    leverage: float = Field(1.0, description="Leverage multiplier", ge=1, le=100)
    price: float = Field(..., description="Entry price", gt=0)
    nbLots: float = Field(..., description="Number of lots/size", gt=0)
    SL: float = Field(..., description="Stop loss price", gt=0)
    TP: Optional[float] = Field(None, description="Take profit price")
    
    @validator('direction')
    def validate_direction(cls, v):
        if v.upper() not in ["LONG", "SHORT"]:
            raise ValueError('Direction must be LONG or SHORT')
        return v.upper()


class RecommendationRequest(BaseModel):
    """Request for trading recommendation"""
    pair: str = Field(..., description="Trading pair, e.g., SOL-USDC")
    chain: str = Field("Solana", description="Blockchain name")
    previous: List[Candle] = Field(..., description="Historical candles")
    positions: List[Position] = Field(default_factory=list, description="Open positions")
    
    @validator('previous')
    def validate_candles(cls, v):
        if len(v) < 4:
            raise ValueError('At least 4 candles required')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 candles allowed')
        return v


class RecommendationResponse(BaseModel):
    """Response with trading recommendation"""
    recommendation: str = Field(..., description="LONG, SHORT, WAIT, or CLOSE")
    confidence: float = Field(..., description="Confidence score 0-1", ge=0, le=1)
    position: Optional[Position] = Field(None, description="Position to close (if recommendation is CLOSE)")
    reasoning: Optional[str] = Field(None, description="Explanation of recommendation")
    features: Optional[Dict[str, Any]] = Field(None, description="ML features used")
    
    @validator('recommendation')
    def validate_recommendation(cls, v):
        valid = ["LONG", "SHORT", "WAIT", "CLOSE"]
        if v.upper() not in valid:
            raise ValueError(f'Recommendation must be one of {valid}')
        return v.upper()


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="ML model status")
    redis_connected: bool = Field(..., description="Redis connection status")
    uptime: float = Field(..., description="Service uptime in seconds")
    requests_processed: int = Field(..., description="Total requests processed")


class MetricsResponse(BaseModel):
    """Metrics response"""
    total_requests: int = Field(..., description="Total requests")
    recommendations_given: Dict[str, int] = Field(..., description="Count by type")
    avg_confidence: float = Field(..., description="Average confidence")
    avg_response_time: float = Field(..., description="Average response time in ms")
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    errors: int = Field(..., description="Total errors")


# ============================================================================
# Dependencies & Configuration
# ============================================================================

# Load configuration
def load_config():
    """Load configuration from YAML"""
    import yaml
    from pathlib import Path
    
    config_path = Path("config/config.yaml")
    default_config = {
        "server": {"host": "0.0.0.0", "port": 8000},
        "ai": {"min_confidence": 0.65},
        "security": {"require_auth": True, "api_key_header": "X-API-Key"},
        "monitoring": {"metrics_enabled": True}
    }
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            default_config.update(config)
    
    # Also check environment variables
    api_keys = os.getenv("API_KEYS", "").split(",")
    default_config["security"]["valid_api_keys"] = [k.strip() for k in api_keys if k.strip()]
    
    return default_config

config = load_config()

# API Key security
api_key_header = APIKeyHeader(name=config["security"]["api_key_header"], auto_error=False)

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Verify API key"""
    if not config["security"]["require_auth"]:
        return True
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    valid_keys = config["security"].get("valid_api_keys", [])
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return True

# ============================================================================
# Application Lifespan
# ============================================================================

# Global state
ml_model: Optional[MLModel] = None
data_store: Optional[DataStore] = None
recommendation_engine: Optional[RecommendationEngine] = None
start_time = datetime.now()
request_count = 0
metrics = {
    "total_requests": 0,
    "recommendations": {"LONG": 0, "SHORT": 0, "WAIT": 0, "CLOSE": 0},
    "response_times": [],
    "cache_hits": 0,
    "cache_misses": 0,
    "errors": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    global ml_model, data_store, recommendation_engine
    
    # Startup
    logger.info("Starting AI Recommendation Service...")
    
    try:
        # Initialize ML model
        ml_model = MLModel(config["ai"]["model_path"])
        await ml_model.load()
        logger.info(f"ML model loaded: {ml_model.is_loaded}")
        
        # Initialize data store
        data_store = DataStore(config["database"]["redis_url"])
        await data_store.connect()
        logger.info("Data store connected")
        
        # Initialize recommendation engine
        recommendation_engine = RecommendationEngine(ml_model, data_store, config)
        logger.info("Recommendation engine initialized")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    logger.info(f"Service started on {config['server']['host']}:{config['server']['port']}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Recommendation Service...")
    
    if data_store:
        await data_store.disconnect()
    
    logger.info("Service shutdown complete")

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="AI Trading Recommendation Service",
    description="Provides AI-powered trading signals for crypto assets",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["security"].get("allowed_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "AI Trading Recommendation Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global start_time
    
    return HealthResponse(
        status="healthy" if ml_model and ml_model.is_loaded else "degraded",
        version="1.0.0",
        model_loaded=ml_model.is_loaded if ml_model else False,
        redis_connected=data_store.is_connected if data_store else False,
        uptime=(datetime.now() - start_time).total_seconds(),
        requests_processed=request_count
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Service metrics endpoint"""
    global metrics
    
    total_cache = metrics["cache_hits"] + metrics["cache_misses"]
    cache_hit_rate = metrics["cache_hits"] / total_cache if total_cache > 0 else 0
    
    avg_response_time = sum(metrics["response_times"]) / len(metrics["response_times"]) if metrics["response_times"] else 0
    
    total_recommendations = sum(metrics["recommendations"].values())
    if total_recommendations > 0:
        weighted_confidences = {
            "LONG": metrics["recommendations"]["LONG"] * 0.7,  # Assuming LONG/SHORT have avg 70% confidence
            "SHORT": metrics["recommendations"]["SHORT"] * 0.7,
            "WAIT": metrics["recommendations"]["WAIT"] * 0.5,  # WAIT has lower confidence
            "CLOSE": metrics["recommendations"]["CLOSE"] * 0.8  # CLOSE has high confidence
        }
        avg_confidence = sum(weighted_confidences.values()) / total_recommendations
    else:
        avg_confidence = 0
    
    return MetricsResponse(
        total_requests=metrics["total_requests"],
        recommendations_given=metrics["recommendations"],
        avg_confidence=avg_confidence,
        avg_response_time=avg_response_time,
        cache_hit_rate=cache_hit_rate,
        errors=metrics["errors"]
    )


@app.post("/api/v1/recommendation", response_model=RecommendationResponse)
async def get_recommendation(
    request: RecommendationRequest,
    auth: bool = Depends(verify_api_key)
):
    """
    Get trading recommendation based on market data
    
    - **pair**: Trading pair (e.g., SOL-USDC)
    - **chain**: Blockchain (e.g., Solana)
    - **previous**: Historical OHLCV candles with indicators
    - **positions**: Currently open positions
    
    Returns recommendation with confidence score
    """
    global request_count, metrics
    
    start_time_req = datetime.now()
    request_count += 1
    metrics["total_requests"] += 1
    
    try:
        logger.info(f"Recommendation request for {request.pair}")
        logger.debug(f"Request data: {request.json()[:200]}...")
        
        # Check cache first
        cache_key = f"recommendation:{request.pair}:{hash(str(request.json()))}"
        cached_result = await data_store.get(cache_key) if data_store else None
        
        if cached_result:
            metrics["cache_hits"] += 1
            logger.info(f"Cache hit for {request.pair}")
            return RecommendationResponse(**json.loads(cached_result))
        
        metrics["cache_misses"] += 1
        
        # Generate recommendation
        if not recommendation_engine:
            raise HTTPException(status_code=503, detail="Recommendation engine not available")
        
        result = await recommendation_engine.generate_recommendation(request)
        
        # Update metrics
        metrics["recommendations"][result.recommendation] += 1
        
        # Cache result (5 minutes)
        if data_store:
            await data_store.set(
                cache_key,
                result.json(),
                ttl=300  # 5 minutes
            )
        
        # Calculate response time
        response_time = (datetime.now() - start_time_req).total_seconds() * 1000
        metrics["response_times"].append(response_time)
        if len(metrics["response_times"]) > 1000:
            metrics["response_times"] = metrics["response_times"][-1000:]
        
        logger.info(f"Recommendation: {result.recommendation} ({result.confidence:.2%}) for {request.pair}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Error processing recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/v1/batch-recommendations")
async def batch_recommendations(
    requests: List[RecommendationRequest],
    auth: bool = Depends(verify_api_key)
):
    """Get multiple recommendations at once"""
    results = []
    
    for request in requests:
        try:
            result = await get_recommendation(request, auth)
            results.append(result)
        except Exception as e:
            results.append({
                "pair": request.pair,
                "error": str(e),
                "recommendation": "ERROR"
            })
    
    return {"results": results}


@app.get("/api/v1/model-info")
async def model_info(auth: bool = Depends(verify_api_key)):
    """Get information about the ML model"""
    if not ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": ml_model.model_type,
        "features": ml_model.feature_names,
        "accuracy": ml_model.accuracy,
        "trained_at": ml_model.trained_at,
        "version": ml_model.version
    }


@app.post("/api/v1/train")
async def train_model(
    training_data: Dict[str, Any],
    auth: bool = Depends(verify_api_key)
):
    """Retrain the ML model (admin endpoint)"""
    # This would require special admin privileges
    # For now, just return mock response
    return {
        "status": "training_started",
        "message": "Model retraining in background",
        "job_id": "train_123"
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle generic exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Trading Recommendation Service")
    parser.add_argument("--host", default=config["server"]["host"], help="Host to bind to")
    parser.add_argument("--port", type=int, default=config["server"]["port"], help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    parser.add_argument("--workers", type=int, default=config["server"].get("workers", 1))
    
    args = parser.parse_args()
    
    uvicorn.run(
        "src.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
