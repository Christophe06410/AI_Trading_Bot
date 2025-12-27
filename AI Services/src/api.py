#!/usr/bin/env python3
"""
AI Service - Production Ready FastAPI Application
Loads models from ML-Pipeline and serves recommendations
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
import uvicorn

# Local imports
from src.ml_predictor import MLPredictor
from src.recommendation_engine import RecommendationEngine
from src.data_cache import DataCache
from src.config_loader import Config

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
# Pydantic Models (from your specification)
# ============================================================================

class Candle(BaseModel):
    timestamp: str = Field(..., description="ISO format timestamp")
    o: float = Field(..., description="Open price", gt=0)
    h: float = Field(..., description="High price", gt=0)
    l: float = Field(..., description="Low price", gt=0)
    c: float = Field(..., description="Close price", gt=0)
    v: float = Field(..., description="Volume", ge=0)
    ema200: Optional[float] = Field(None, description="200-period EMA")

class Position(BaseModel):
    id: Optional[str] = Field(None, description="Position ID")
    timestamp: str = Field(..., description="Entry time ISO format")
    direction: str = Field(..., description="LONG or SHORT")
    leverage: float = Field(1.0, description="Leverage multiplier", ge=1, le=100)
    price: float = Field(..., description="Entry price", gt=0)
    nbLots: float = Field(..., description="Number of lots/size", gt=0)
    SL: float = Field(..., description="Stop loss price", gt=0)
    TP: Optional[float] = Field(None, description="Take profit price")

class RecommendationRequest(BaseModel):
    pair: str = Field(..., description="Trading pair, e.g., BTC-USDT")
    chain: str = Field("Solana", description="Blockchain name")
    previous: List[Candle] = Field(..., description="Historical candles")
    positions: List[Position] = Field(default_factory=list, description="Open positions")

class RecommendationResponse(BaseModel):
    recommendation: str = Field(..., description="LONG, SHORT, WAIT, or CLOSE")
    confidence: float = Field(..., description="Confidence score 0-1", ge=0, le=1)
    position: Optional[Position] = Field(None, description="Position to close")
    reasoning: Optional[str] = Field(None, description="Explanation")
    model_used: Optional[str] = Field(None, description="Which ML model was used")

# ============================================================================
# FastAPI Application
# ============================================================================

# Load configuration
config = Config().load()

# Initialize services
ml_predictor = MLPredictor(config)
data_cache = DataCache(config)
recommendation_engine = RecommendationEngine(ml_predictor, data_cache, config)

# API Key security
api_key_header = APIKeyHeader(name=config.security.api_key_header, auto_error=False)

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Verify API key"""
    if not config.security.require_auth:
        return True
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    valid_keys = config.security.valid_api_keys
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBEDDEN,
            detail="Invalid API key"
        )
    
    return True

# Create FastAPI app
app = FastAPI(
    title="AI Trading Recommendation Service",
    description="Provides AI-powered trading signals using ML-Pipeline models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting AI Service...")
    
    # Initialize ML predictor (loads models from ML-Pipeline)
    await ml_predictor.initialize()
    
    # Initialize cache
    await data_cache.initialize()
    
    logger.info(f"AI Service started on port {config.server.port}")
    logger.info(f"Loaded {len(ml_predictor.models)} ML models from ML-Pipeline")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ai-service",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": ml_predictor.is_initialized,
        "cache_connected": data_cache.is_connected
    }

# Main recommendation endpoint
@app.post("/api/v1/recommendation", response_model=RecommendationResponse)
async def get_recommendation(
    request: RecommendationRequest,
    auth: bool = Depends(verify_api_key)
):
    """
    Get trading recommendation using ML-Pipeline models
    
    - **pair**: Trading pair (e.g., BTC-USDT)
    - **chain**: Blockchain (e.g., Solana)
    - **previous**: Historical OHLCV candles with indicators
    - **positions**: Currently open positions
    
    Returns recommendation with confidence score
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Recommendation request for {request.pair}")
        
        # Check cache first
        cache_key = f"recommendation:{request.pair}:{hash(str(request.dict()))}"
        cached_result = await data_cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for {request.pair}")
            return RecommendationResponse(**cached_result)
        
        # Generate recommendation using ML models
        result = await recommendation_engine.generate_recommendation(request)
        
        # Cache result
        await data_cache.set(cache_key, result.dict(), ttl=300)
        
        # Log performance
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            f"Recommendation generated: {result.recommendation} "
            f"({result.confidence:.2%}) in {processing_time:.1f}ms"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Batch recommendations endpoint
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
            results.append(result.dict())
        except Exception as e:
            results.append({
                "pair": request.pair,
                "error": str(e),
                "recommendation": "ERROR"
            })
    
    return {"results": results}

# Model info endpoint
@app.get("/api/v1/model-info")
async def model_info(auth: bool = Depends(verify_api_key)):
    """Get information about loaded ML models"""
    models_info = []
    
    for model_name, model in ml_predictor.models.items():
        models_info.append({
            "name": model_name,
            "type": model.model_type,
            "accuracy": model.accuracy,
            "features": model.feature_names,
            "loaded_at": model.loaded_at
        })
    
    return {
        "total_models": len(models_info),
        "models": models_info,
        "ml_pipeline_path": str(ml_predictor.model_path)
    }

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Service...")
    await data_cache.close()

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Trading Recommendation Service")
    parser.add_argument("--host", default=config.server.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.server.port, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=config.server.workers
    )

if __name__ == "__main__":
    main()
