"""
AI Service client for getting trading recommendations - Updated
"""

import aiohttp
import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.config import TradingBotConfig
from src.utils import get_logger, retry_with_backoff

logger = get_logger(__name__)


@dataclass
class AIRecommendation:
    """AI trading recommendation"""
    id: str
    pair: str
    direction: str  # "LONG", "SHORT", "WAIT", "CLOSE"
    confidence: float
    position_id: Optional[str] = None  # For CLOSE recommendations
    reasoning: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pair": self.pair,
            "direction": self.direction,
            "confidence": self.confidence,
            "position_id": self.position_id,
            "reasoning": self.reasoning,
            "features": self.features,
            "timestamp": self.timestamp.isoformat()
        }


class AIClient:
    """Client for AI Service API with improved error handling"""
    
    def __init__(self, config: TradingBotConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = config.get_full_ai_endpoint()
        self.api_key = None
        self.health_status = False
        self.last_success = None
        
        # Load API key from environment
        import os
        self.api_key = os.getenv("AI_API_KEY")
        
        self.logger = get_logger(__name__)
    
    async def initialize(self):
        """Initialize AI client"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.ai_service.timeout),
            headers={
                "User-Agent": "TradingBot/3.0.0",
                "Accept": "application/json"
            }
        )
        
        # Test connection
        self.health_status = await self.check_connection()
        
        if self.health_status:
            self.logger.info(f"AI client initialized for {self.base_url}")
        else:
            self.logger.warning(f"AI client initialized but connection test failed")
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    async def get_recommendations(
        self,
        market_data: Dict[str, Dict[str, Any]],
        positions: Optional[List[Dict[str, Any]]] = None
    ) -> List[AIRecommendation]:
        """Get trading recommendations from AI service"""
        try:
            # Prepare request
            request_data = self._prepare_request_data(market_data, positions)
            
            # Make request
            response = await self._make_request(request_data)
            
            # Parse response
            recommendations = self._parse_response(response)
            
            # Update health status
            self.health_status = True
            self.last_success = datetime.now()
            
            self.logger.debug(f"Got {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get recommendations: {e}")
            self.health_status = False
            return []
    
    def _prepare_request_data(
        self,
        market_data: Dict[str, Dict[str, Any]],
        positions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Prepare request data for AI service"""
        import time
        
        # For each pair, prepare candle data
        candles_by_pair = {}
        
        for pair, data in market_data.items():
            # Create mock candles based on current price
            # In production, you would use real historical candles
            current_price = data.get("price", 0)
            
            candles = []
            for i in range(4):  # Last 4 candles
                timestamp = datetime.now().timestamp() - (i * 300)  # 5-minute intervals
                candle = {
                    "timestamp": datetime.fromtimestamp(timestamp).isoformat() + "Z",
                    "o": current_price * (0.99 if i % 2 == 0 else 1.01),
                    "h": current_price * 1.02,
                    "l": current_price * 0.98,
                    "c": current_price,
                    "v": 1000.0 + (i * 100),
                    "ema200": current_price * 0.95
                }
                candles.append(candle)
            
            candles_by_pair[pair] = candles
        
        # Use first pair for request (simplified)
        pair = list(market_data.keys())[0] if market_data else "SOL/USDC"
        
        return {
            "pair": pair.replace("/", "-"),
            "chain": "Solana",
            "previous": candles_by_pair.get(pair, []),
            "positions": positions or []
        }
    
    async def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to AI service"""
        if not self.session:
            raise Exception("AI client not initialized")
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key if available
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.session.post(
                self.base_url,
                json=data,
                headers=headers
            ) as response:
                
                response_time = asyncio.get_event_loop().time() - start_time
                
                if response.status == 200:
                    self.logger.debug(f"AI request successful: {response_time:.2f}s")
                    return await response.json()
                else:
                    error_text = await response.text()
                    self.logger.error(
                        f"AI service error {response.status}: {error_text}",
                        response_time=response_time
                    )
                    raise Exception(f"AI service error: {response.status}")
                    
        except asyncio.TimeoutError:
            self.logger.error("AI service timeout")
            raise Exception("AI service timeout")
        except aiohttp.ClientError as e:
            self.logger.error(f"AI service connection error: {e}")
            raise Exception(f"Connection error: {e}")
    
    def _parse_response(self, response: Dict[str, Any]) -> List[AIRecommendation]:
        """Parse AI service response"""
        import uuid
        
        recommendations = []
        
        # Handle single recommendation
        if "recommendation" in response:
            rec_id = str(uuid.uuid4())[:8]
            
            # Parse pair from response or use default
            pair = response.get("pair", "SOL-USDC").replace("-", "/")
            
            rec = AIRecommendation(
                id=rec_id,
                pair=pair,
                direction=response.get("recommendation", "WAIT").upper(),
                confidence=float(response.get("confidence", 0.5)),
                position_id=response.get("position", {}).get("id") if response.get("position") else None,
                reasoning=response.get("reasoning"),
                features=response.get("features"),
                timestamp=datetime.now()
            )
            recommendations.append(rec)
        
        # Handle batch recommendations
        elif "results" in response:
            for result in response["results"]:
                rec_id = str(uuid.uuid4())[:8]
                
                rec = AIRecommendation(
                    id=rec_id,
                    pair=result.get("pair", "SOL/USDC"),
                    direction=result.get("recommendation", "WAIT").upper(),
                    confidence=float(result.get("confidence", 0.5)),
                    position_id=result.get("position_id"),
                    reasoning=result.get("reasoning"),
                    features=result.get("features"),
                    timestamp=datetime.now()
                )
                recommendations.append(rec)
        
        # Filter by minimum confidence
        min_confidence = self.config.ai_service.min_confidence
        recommendations = [
            r for r in recommendations 
            if r.confidence >= min_confidence
        ]
        
        # Log recommendations
        for rec in recommendations:
            self.logger.debug(
                f"Recommendation: {rec.direction} {rec.pair} "
                f"(confidence: {rec.confidence:.2%})"
            )
        
        return recommendations
    
    async def check_connection(self) -> bool:
        """Check connection to AI service"""
        if not self.session:
            return False
        
        try:
            # Try health endpoint first
            health_url = f"{self.config.ai_service.url}/health"
            
            async with self.session.get(
                health_url,
                timeout=5
            ) as response:
                if response.status == 200:
                    return True
            
            # Fallback to root endpoint
            async with self.session.get(
                self.config.ai_service.url,
                timeout=5
            ) as response:
                return response.status < 500
                
        except Exception as e:
            self.logger.debug(f"AI service connection check failed: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            "connected": self.health_status,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key)
        }
    
    async def close(self):
        """Close AI client"""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("AI client closed")
