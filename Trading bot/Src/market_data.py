"""
Market data fetching and caching
"""

import aiohttp
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from src.config import TradingBotConfig
from src.utils import get_logger, Cache

logger = get_logger(__name__)


class MarketData:
    """Fetches and caches market data"""
    
    def __init__(self, config: TradingBotConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = Cache(ttl=30)  # 30 second cache
        
        # Price sources in order of preference
        self.price_sources = [
            self._get_price_jupiter,
            self._get_price_coingecko,
            self._get_price_birdeye
        ]
        
        self.logger = get_logger(__name__)
    
    async def initialize(self):
        """Initialize market data service"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        self.logger.info("Market data service initialized")
    
    async def get_price(self, pair: str) -> Optional[float]:
        """Get current price for trading pair"""
        # Check cache first
        cached = self.cache.get(f"price:{pair}")
        if cached is not None:
            return cached
        
        # Try each price source
        price = None
        for source in self.price_sources:
            try:
                price = await source(pair)
                if price and price > 0:
                    break
            except Exception as e:
                self.logger.debug(f"Price source failed: {e}")
                continue
        
        # Cache the result
        if price:
            self.cache.set(f"price:{pair}", price)
            self.logger.debug(f"Price for {pair}: ${price:.4f}")
        
        return price
    
    async def _get_price_jupiter(self, pair: str) -> Optional[float]:
        """Get price from Jupiter API"""
        # Parse pair (e.g., "SOL/USDC")
        try:
            base, quote = pair.split("/")
            
            # Map to Jupiter token IDs
            token_map = {
                "SOL": "SOL",
                "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
                "SRM": "SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt"
            }
            
            token_id = token_map.get(base, base)
            
            url = f"{self.config.solana.dex.jupiter_api_url}/price"
            params = {"ids": token_id}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data["data"][token_id]["price"])
                else:
                    return None
                    
        except Exception as e:
            self.logger.warning(f"Jupiter price failed: {e}")
            return None
    
    async def _get_price_coingecko(self, pair: str) -> Optional[float]:
        """Get price from CoinGecko (fallback)"""
        # Map pairs to CoinGecko IDs
        cg_map = {
            "SOL/USDC": "solana",
            "RAY/USDC": "raydium",
            "SRM/USDC": "serum"
        }
        
        cg_id = cg_map.get(pair)
        if not cg_id:
            return None
        
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": cg_id,
                "vs_currencies": "usd"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data[cg_id]["usd"])
                else:
                    return None
                    
        except Exception:
            return None
    
    async def _get_price_birdeye(self, pair: str) -> Optional[float]:
        """Get price from BirdEye (requires API key)"""
        # This would require an API key
        return None
    
    async def get_candles(
        self,
        pair: str,
        timeframe: str = "5m",
        limit: int = 100
    ) -> Optional[list]:
        """Get historical candles"""
        cache_key = f"candles:{pair}:{timeframe}:{limit}"
        
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # For now, return mock candles
        # In production, implement actual candle fetching
        candles = self._generate_mock_candles(pair, timeframe, limit)
        
        self.cache.set(cache_key, candles, ttl=300)  # Cache for 5 minutes
        return candles
    
    def _generate_mock_candles(
        self,
        pair: str,
        timeframe: str,
        limit: int
    ) -> list:
        """Generate mock candles for testing"""
        import random
        import time
        
        candles = []
        base_price = 100.0  # Starting price
        
        for i in range(limit):
            timestamp = datetime.now() - timedelta(minutes=i * 5)
            
            # Generate random OHLCV
            open_price = base_price + random.uniform(-2, 2)
            close_price = open_price + random.uniform(-1, 1)
            high_price = max(open_price, close_price) + random.uniform(0, 1)
            low_price = min(open_price, close_price) - random.uniform(0, 1)
            volume = random.uniform(1000, 10000)
            
            candle = {
                "timestamp": timestamp.isoformat() + "Z",
                "open": round(open_price, 4),
                "high": round(high_price, 4),
                "low": round(low_price, 4),
                "close": round(close_price, 4),
                "volume": round(volume, 2)
            }
            
            candles.append(candle)
            base_price = close_price
        
        return candles
    
    async def get_volatility(self, pair: str, period: int = 20) -> float:
        """Calculate price volatility"""
        try:
            candles = await self.get_candles(pair, "5m", period)
            if not candles:
                return 0.0
            
            # Calculate standard deviation of returns
            returns = []
            for i in range(1, len(candles)):
                prev = candles[i-1]["close"]
                curr = candles[i]["close"]
                if prev > 0:
                    returns.append((curr - prev) / prev)
            
            if not returns:
                return 0.0
            
            import statistics
            volatility = statistics.stdev(returns) * 100  # Convert to percentage
            return round(volatility, 2)
            
        except Exception as e:
            self.logger.warning(f"Volatility calculation failed: {e}")
            return 0.0
    
    async def close(self):
        """Close market data service"""
        if self.session:
            await self.session.close()
            self.logger.info("Market data service closed")
