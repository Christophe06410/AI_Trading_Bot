"""
Utility functions for trading bot
"""

import logging
import structlog
import time
import hashlib
import base64
import json
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import asyncio

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


def get_logger(name: str = "trading_bot") -> structlog.BoundLogger:
    """Get structured logger"""
    return structlog.get_logger(name)


def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration"""
    log_level = config.get("level", "INFO").upper()
    
    # Set standard logging level
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(message)s",
        handlers=[
            logging.FileHandler(config.get("file", "logs/trading_bot.log")),
            logging.StreamHandler()
        ]
    )


class Cache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl: int = 300):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Any:
        """Get value from cache"""
        if key in self.cache:
            value, expiry = self.cache[key]
            if time.time() < expiry:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        ttl = ttl or self.ttl
        expiry = time.time() + ttl
        self.cache[key] = (value, expiry)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """Retry decorator with exponential backoff"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= backoff_factor
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def encrypt_data(data: bytes, key: str) -> bytes:
    """Encrypt data using simple XOR (for demonstration)"""
    # In production, use proper encryption like AES
    key_bytes = key.encode()[:len(data)]
    encrypted = bytes(a ^ b for a, b in zip(data, key_bytes))
    return base64.b64encode(encrypted)


def decrypt_data(encrypted_data: bytes, key: str) -> bytes:
    """Decrypt data"""
    encrypted = base64.b64decode(encrypted_data)
    key_bytes = key.encode()[:len(encrypted)]
    return bytes(a ^ b for a, b in zip(encrypted, key_bytes))


def generate_position_id(pair: str, direction: str) -> str:
    """Generate unique position ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_hash = hashlib.md5(f"{pair}_{direction}_{timestamp}".encode()).hexdigest()[:8]
    return f"{pair.replace('/', '_')}_{direction}_{timestamp}_{unique_hash}"


def format_price(price: float) -> str:
    """Format price for display"""
    if price >= 1000:
        return f"${price:,.0f}"
    elif price >= 1:
        return f"${price:,.2f}"
    else:
        return f"${price:.4f}"


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    size: float,
    direction: str
) -> float:
    """Calculate PnL"""
    if direction == "LONG":
        return (exit_price - entry_price) * size
    else:  # SHORT
        return (entry_price - exit_price) * size


def calculate_pnl_percent(
    entry_price: float,
    exit_price: float,
    direction: str
) -> float:
    """Calculate PnL percentage"""
    if direction == "LONG":
        return ((exit_price - entry_price) / entry_price) * 100
    else:  # SHORT
        return ((entry_price - exit_price) / entry_price) * 100


def is_first_friday() -> bool:
    """Check if today is first Friday of month"""
    today = datetime.now()
    return today.weekday() == 4 and today.day <= 7


def validate_pair_format(pair: str) -> bool:
    """Validate trading pair format"""
    return "/" in pair and len(pair.split("/")) == 2


def parse_pair(pair: str) -> tuple:
    """Parse trading pair into base and quote"""
    if "/" in pair:
        return pair.split("/")
    elif "-" in pair:
        return pair.split("-")
    else:
        raise ValueError(f"Invalid pair format: {pair}")


def log_execution_time(func: Callable):
    """Decorator to log function execution time"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            logger = get_logger(func.__module__)
            logger.debug(
                f"Function {func.__name__} executed",
                execution_time=execution_time
            )
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            execution_time = time.time() - start_time
            logger = get_logger(func.__module__)
            logger.debug(
                f"Function {func.__name__} executed",
                execution_time=execution_time
            )
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
