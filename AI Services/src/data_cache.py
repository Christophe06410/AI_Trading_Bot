"""
Redis Data Cache for AI Service
"""

import json
import asyncio
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataCache:
    """Redis cache for AI Service"""
    
    def __init__(self, config):
        self.config = config
        self.redis = None
        self.is_connected = False
    
    async def initialize(self):
        """Initialize Redis connection"""
        if not self.config.cache.enabled:
            logger.info("Cache disabled")
            return
        
        try:
            import redis.asyncio as redis
            
            self.redis = redis.from_url(
                self.config.cache.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            await self.redis.ping()
            self.is_connected = True
            logger.info(f"Connected to Redis at {self.config.cache.redis_url}")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self._memory_cache = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis and self.is_connected:
                value = await self.redis.get(key)
                return json.loads(value) if value else None
            elif hasattr(self, '_memory_cache'):
                return self._memory_cache.get(key)
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        try:
            if self.redis and self.is_connected:
                await self.redis.setex(key, ttl, json.dumps(value))
            elif hasattr(self, '_memory_cache'):
                self._memory_cache[key] = value
                # Simple TTL simulation
                asyncio.create_task(self._remove_after_ttl(key, ttl))
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
    
    async def _remove_after_ttl(self, key: str, ttl: int):
        """Remove key from memory cache after TTL"""
        await asyncio.sleep(ttl)
        if hasattr(self, '_memory_cache') and key in self._memory_cache:
            del self._memory_cache[key]
    
    async def close(self):
        """Close Redis connection"""
        if self.redis and self.is_connected:
            await self.redis.close()
            self.is_connected = False
            logger.info("Redis connection closed")